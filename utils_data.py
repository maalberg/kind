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


class dataset_factory:
    def create_dataset(self, args):

        if args.dataset_id=='kind-detuning-filter':
            return dataset_detuning_filter(
                args.file_dir, args.file_name, args.file_ext,
                args.data_nsample,
                (args.data_train_size, args.data_test_size),
                args.batch_size, (args.lookback_nsample, args.forecast_nsample))

        if args.dataset_id=='kind-detuning-meas':
            return dataset_detuning_meas(
                args.file_dir, args.file_name, args.file_ext,
                args.data_nsample,
                (args.data_train_size, args.data_test_size),
                args.batch_size, (args.lookback_nsample, args.forecast_nsample))

        if args.dataset_id=='kind-detuning-raw':
            return dataset_detuning_raw(
                args.file_dir, args.file_name, args.file_ext,
                args.data_nsample,
                (args.data_train_size, args.data_test_size),
                args.batch_size, (args.lookback_nsample, args.forecast_nsample))

        else:
            return


class dataset(interface):

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample, data_split_size, batch_size, window_nsample):
        self.file_dir = file_dir
        self.file_name = file_name
        self.file_ext = file_ext
        self.data_nsample = data_nsample
        self.split_size = data_split_size
        self.batch_size = batch_size
        self.window_nsample = window_nsample

    def load(self, data_type='stat'):
        assert data_type in ['stat', 'trans', 'mixed']

        # --! read rolling windows from a data file
        if data_type=='mixed':
            # --! read both data types
            timeseries_stat = self.read_timeseries(self.make_path(data_type='stat'))
            timeseries_trans = self.read_timeseries(self.make_path(data_type='trans'))

            # --! update normalization statistics
            self.init_normalization(torch.cat([timeseries_stat, timeseries_trans]))

            window_stat = self.extract_window(timeseries_stat)
            window_trans = self.extract_window(timeseries_trans)

            # --! ensure both data have the same size in the first dimension
            nwindow = window_stat.shape[0] if window_stat.shape[0] < window_trans.shape[0] else window_trans.shape[0]
            window_stat = window_stat[:nwindow]
            window_trans = window_trans[:nwindow]

            # --! interleave both data to lay out windows as stationary, transient, stationary, transient, etc.
            window = torch.stack([window_stat, window_trans], dim=1)
            window = torch.flatten(window, start_dim=0, end_dim=1)
        else:
            timeseries = self.read_timeseries(self.make_path(data_type))

            # --! update normalization statistics
            self.init_normalization(timeseries)

            window = self.extract_window(timeseries)

        # --! adapt control mask in read data windows to comply with current dataset use case
        window = self.adapt_mask(window)
        window = self.noise(window)

        # --! split data into train, valid and test partitions
        train_data, valid_test_data = train_test_split(window, train_size=self.split_size[0], shuffle=True)
        valid_data, test_data = train_test_split(valid_test_data, test_size=self.split_size[1], shuffle=True)

        # --! normalize training data
        train_data = self.normalize(train_data)

        # --! create datasets by splitting the windows into lookback and forecast parts
        train_back, train_fore = torch.split(train_data, list(self.window_nsample), dim=1)
        valid_back, valid_fore = torch.split(valid_data, list(self.window_nsample), dim=1)
        test_back, test_fore = torch.split(test_data, list(self.window_nsample), dim=1)

        # --! since our datasets are already tensors, then wrap them in tensor datasets
        train_dataset = torch.utils.data.TensorDataset(train_back, train_fore)
        valid_dataset = torch.utils.data.TensorDataset(valid_back, valid_fore)
        test_dataset = torch.utils.data.TensorDataset(test_back, test_fore)

        # --! wrap the datasets into loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

    @abstractmethod
    def make_path(self, data_type='stat'):
        return

    @abstractmethod
    def extract_target(self, window):
        """ Extracts target dimension from given ``window``. """
        return

    @abstractmethod
    def adapt_mask(self, window):
        """ Adapts mask data dimension in ``window`` to the use in current dataset. """
        return

    def read_timeseries(self, path):
        """ Reads time series from a data file located at ``path``. """

        # --! read data from a csv file
        data = self.read_csv(path)

        # --! convert read data to a 3D torch tensor where the first dimension contains time series
        ntimeseries = data.shape[0] // self.data_nsample
        return torch.reshape(data, (ntimeseries, self.data_nsample, data.shape[1]))

    @abstractmethod
    def init_normalization(self, timeseries):
        return

    @abstractmethod
    def normalize(self, window):
        return

    @abstractmethod
    def denormalize(self, window):
        return

    @abstractmethod
    def noise(self, window):
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


class dataset_detuning_raw(dataset):

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='stat'):
        file_name = self.file_name + '_raw' + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        return window[:, :, [0]]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def init_normalization(self, timeseries):

        detuning_control, mask = torch.split(timeseries, [2, 1], dim=-1)

        # --! take global statistics
        self.mean = detuning_control.mean()
        self.std = detuning_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==1:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = torch.clip((detuning - self.mean) / self.std, min=-3.0, max=3.0)
            control = torch.clip((control - self.mean) / self.std, min=-3.0, max=3.0)
            control = control * mask

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==1:
            window = window * self.std + self.mean
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = detuning * self.std + self.mean
            control = control * self.std + self.mean

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def noise(self, window):
        return window


class dataset_detuning_filter(dataset):

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='stat'):
        file_name = self.file_name + '_stat' + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        return window[:, :, [0]]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def init_normalization(self, timeseries):

        detuning_control, mask = torch.split(timeseries, [2, 1], dim=-1)

        # --! take global statistics
        self.mean = detuning_control.mean()
        self.std = detuning_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==1:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = torch.clip((detuning - self.mean) / self.std, min=-3.0, max=3.0)
            control = torch.clip((control - self.mean) / self.std, min=-3.0, max=3.0)
            control = control * mask

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==1:
            window = window * self.std + self.mean
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = detuning * self.std + self.mean
            control = control * self.std + self.mean

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def noise(self, window):
        detuning, control, mask = torch.split(window, 1, dim=-1)
        detuning = detuning + 1e-3 * torch.randn(detuning.shape)

        return torch.cat([detuning, control, mask], dim=-1)


class dataset_detuning_meas(dataset):

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='stat'):
        file_name = self.file_name + '_' + data_type + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        return window[:, :, [0]]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def extract_window(self, timeseries):

        # --! given time series should already contain windows
        return timeseries

    def init_normalization(self, timeseries):

        detuning_control, mask = torch.split(timeseries, [2, 1], dim=-1)

        # --! take global statistics
        self.mean = detuning_control.mean()
        self.std = detuning_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==1:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = torch.clip((detuning - self.mean) / self.std, min=-3.0, max=3.0)
            control = torch.clip((control - self.mean) / self.std, min=-3.0, max=3.0)
            control = control * mask

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==1:
            window = window * self.std + self.mean
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = detuning * self.std + self.mean
            control = control * self.std + self.mean

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def noise(self, window):

        # --! measured data already has noise
        return window


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


def read_datafile(name: str, datachunk_len) -> torch.Tensor:
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
        np.loadtxt(name + '.csv', delimiter=',', dtype=np.float32, ndmin=2))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))


def write_datafile(name: str, data, delim: str = ','):
    """Writes ``data`` to a file named ``name``. The file is written using a comma-separated-value format."""
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt(name + '.csv', filedata, fmt='%.14f', delimiter=delim)

