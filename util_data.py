# --!--------------------------------------------------------------!
# --! utilities for data operations
# --!--------------------------------------------------------------!

from abc import abstractmethod
from abc import ABC as interface

import os
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class normalizer(interface):

    # --! minimum standard deviation to avoid division by zero-like deviation values
    std_min = torch.tensor(1e-3, dtype=torch.float32)

    @abstractmethod
    def normalize(self, timeseries):
        return

    @abstractmethod
    def denormalize(self, timeseries):
        return


class dataset(interface):

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self, args, setpoint, load_normalized=True, extract_windows=True):

        self.args = args
        self.split_size = (args.data_train_size, args.data_test_size)
        self.window_nsample = (args.back_nsample, args.fore_nsample)
        self.setpoint = torch.unsqueeze(torch.unsqueeze(setpoint, 0), 0) # <-- converting a 1d tensor into a 3d tensor
        self.load_normalized = load_normalized
        self.extract_windows = extract_windows

        # --! initialize a normalizer
        self.normalizer = self.init_normalization()

    def load(self, data_type='nom'):
        assert data_type in ['nom', 'exc', 'mixed']

        # --! read rolling windows from a data file
        if data_type=='mixed':
            train_data_nom, valid_data_nom, test_data_nom = self._load_train_test_split(data_type='nom')
            train_data_exc, valid_data_exc, test_data_exc = self._load_train_test_split(data_type='exc')

            train_data = self._mix_data(train_data_nom, train_data_exc)
            valid_data = self._mix_data(valid_data_nom, valid_data_exc)
            test_data = self._mix_data(test_data_nom, test_data_exc)
        else:
            train_data, valid_data, test_data = self._load_train_test_split(data_type)

        # --! if requested, normalize data
        if self.load_normalized:

            train_data = self.normalizer.normalize(train_data)
            valid_data = self.normalizer.normalize(valid_data)
            test_data = self.normalizer.normalize(test_data)

        train_loader = self._create_data_loader(train_data, shuffle=True)
        valid_loader = self._create_data_loader(valid_data, shuffle=False)
        test_loader = self._create_data_loader(test_data, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _load_train_test_split(self, data_type='nom'):

        # --! this method is not supposed to be called for mixed data
        assert data_type in ['nom', 'exc']

        data_nsample = self.args.data_nsample_nom if data_type=='nom' else self.args.data_nsample_exc

        timeseries = self.read_timeseries(self.make_path(data_type), data_nsample)
        if self.extract_windows:
            window = self.extract_window(timeseries)
        else:
            window = timeseries

        # --! split loaded windows into train, valid, test sets of data
        train_data, valid_test_data = train_test_split(window, train_size=self.split_size[0], shuffle=True)
        valid_data, test_data = train_test_split(valid_test_data, test_size=self.split_size[1], shuffle=True)

        return train_data, valid_data, test_data

    def _mix_data(self, data_nom, data_exc):

        if data_nom.shape[0] > data_exc.shape[0]:
            k = int(np.ceil(data_nom.shape[0] / data_exc.shape[0]))
            data_exc = torch.tile(data_exc, (k, 1, 1))
        elif data_exc.shape[0] > data_nom.shape[0]:
            k = int(np.ceil(data_exc.shape[0] / data_nom.shape[0]))
            data_nom = torch.tile(data_nom, (k, 1, 1))

        # --! ensure both data have the same size in the first dimension
        ndata = data_nom.shape[0] if data_nom.shape[0] < data_exc.shape[0] else data_exc.shape[0]
        data_nom = data_nom[:ndata]
        data_exc = data_exc[:ndata]

        # --! interleave both data to lay out windows as nominal, excursion, nominal, excursion, etc.
        data = torch.stack([data_nom, data_exc], dim=1)
        data = torch.flatten(data, start_dim=0, end_dim=1)

        return data

    def _create_data_loader(self, data, shuffle=False):

        # --! create datasets by splitting the windows into lookback and forecast parts
        data_back, data_fore = torch.split(data, list(self.window_nsample), dim=1)

        # --! since our datasets are already tensors, then wrap them in tensor datasets
        dataset = torch.utils.data.TensorDataset(data_back, data_fore)

        # --! wrap the datasets into loaders
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle)

    @abstractmethod
    def make_path(self, data_type='nom'):
        return

    @abstractmethod
    def extract_target(self, window):
        """ Extracts target dimension from given ``window``. """
        return

    def read_timeseries(self, path, data_nsample):
        """ Reads time series from a data file located at ``path``. """

        # --! read data from a csv file
        data = self.read_csv(path)

        # --! convert read data to a 3D torch tensor where the first dimension contains time series
        ntimeseries = data.shape[0] // data_nsample
        return torch.reshape(data, (ntimeseries, data_nsample, data.shape[1]))

    @abstractmethod
    def init_normalization(self):
        return

    def extract_window(self, timeseries):
        """ Extracts windows from ``timeseries`` in a rolling window manner. """

        # --! prepare to extract data windows
        window_nsample = self.window_nsample[0] + self.window_nsample[1]
        data_start = 0
        data_end = timeseries.shape[1] - window_nsample
        windows = []

        # --! extract windows in a rolling window manner
        for ts in timeseries:
            for j in range(data_start, data_end):
                window_start = j
                window_end = window_start + window_nsample
                window = ts[window_start:window_end]

                windows.append(window)
        return torch.stack(windows, dim=0)

    def read_csv(self, data_path):

        # --! read a csv-file with no header
        dataframe = pd.read_csv(data_path, header=None, dtype=np.float32)
        return torch.from_numpy(dataframe.to_numpy())


class minmax_scaler:
    """ Defines a differentiable min-max scaler class suitable to be used during torch-based training. """
    def __init__(self, feature_range=(-1, 1)):
        self.min = None
        self.max = None
        self.min_scaled, self.max_scaled = feature_range

    def fit_transform(self, data, dim=1):

        # --! remember min and max values of given data
        self.min = data.min(dim, keepdim=True)[0]
        self.max = data.max(dim, keepdim=True)[0]

        # --! transform given data according to scaling range
        scale = (self.max_scaled - self.min_scaled) / (self.max - self.min + 1e-8)
        return self.min_scaled + (data - self.min) * scale

    def inverse_transform(self, data):
        scale = (self.max - self.min + 1e-8) / (self.max_scaled - self.min_scaled)
        return self.min + (data - self.min_scaled) * scale


def conv_str2ints(string):
    """ Converts a given comma-separated ``string`` of integers into a list of integers. """
    return [int(item) for item in string.split(',')]


def read_datafile(name: str, datachunk_len, delim=',') -> torch.Tensor:
    """
    Reads data from a file called ``name`` and formats the data based on ``datachunk_len``,
    i.e. the length of one contiguous chunk of data. The file data are expected to be
    in format [T, C], such that the read data could be formatted as [B, T, C],
    where B, T and C are the number of batches, time steps and data channels,
    repectively.
    """

    # --! note that we force numpy loadtxt to return at least a two-dimensional array
    # --! by setting ndmin=2
    data = torch.tensor(
        np.loadtxt(name + '.csv', delimiter=delim, dtype=np.float32, ndmin=2))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))


def write_datafile(name: str, data, delim: str = ','):
    """Writes ``data`` to a file named ``name``. The file is written using a comma-separated-value format."""
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt(name + '.csv', filedata, fmt='%.14f', delimiter=delim)


def ceil(data, decimals=1):
    mul = 10 ** decimals
    return torch.ceil(data * mul) / mul

