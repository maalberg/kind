# --!--------------------------------------------------------------!
# --! utilities for data operations
# --!--------------------------------------------------------------!

from abc import abstractmethod
from abc import ABC as interface

import torch
import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import acf

from matplotlib import pyplot as plt


class dataset_factory:
    def create_dataset(self, args):
        if args.data_t == 'sim':
            return dataset_sim(args.data_path, args.data_nsample)
        else:
            return


class dataset(interface):

    @abstractmethod
    def load(self, window_nsample):
        return

    def read_csv(self, data_path, data_nsample):

        # --! read a csv-file with no header
        dataframe = pd.read_csv(data_path, header=None)
        data      = torch.from_numpy(dataframe.to_numpy())

        # --! return 3D data structure
        ndata     = data.shape[0] // data_nsample
        return torch.reshape(data, (ndata, data_nsample, data.shape[1]))

class dataset_sim(dataset):

    def __init__(self, data_path, data_nsample):
        super().__init__()

        self.data = self.read_csv(data_path, data_nsample)

    def load(self, window_nsample):
        data_start = 0
        data_end   = self.data.shape[1] - window_nsample

        windows = []

        for j in range(data_start, data_end):
            window_start = j
            window_end   = window_start + window_nsample
            window       = self.data[:, window_start:window_end]

            # --! save timeseries framed by current window
            for timeseries in window:
                windows.append(timeseries)

        return torch.stack(windows, dim=0)


class minmax_scaler:
    def __init__(self, feature_range=(-1, 1)):
        self.min = None
        self.max = None
        self.scale_min, self.scale_max = feature_range

    def fit_transform(self, timeseries, dim=1):

        # --! remember min and max values of given timeseries
        self.min = timeseries.min(dim=dim, keepdim=True)[0]
        self.max = timeseries.max(dim=dim, keepdim=True)[0]

        # --! transform given timeseries according to scaling range
        scale = (self.scale_max - self.scale_min) / (self.max - self.min + 1e-8)
        return self.scale_min + (timeseries - self.min) * scale

    def inverse_transform(self, timeseries):
        scale = (self.max - self.min + 1e-8) / (self.scale_max - self.scale_min)
        return self.min + (timeseries - self.scale_min) * scale


def forecastability(x, n_fft=None):
    """
    Calculate the forecastability of a time series using the entropy of its normalized power spectrum.
    Reference: Goerg (2013)

    Forecastability = 1 - H / log(N)
    where H is the Shannon entropy of the normalized power spectrum.
    """
    x = np.asarray(x)
    x = x - np.mean(x)  # remove mean
    
    # Compute the power spectrum
    n_fft = n_fft or len(x)
    fft = np.fft.fft(x, n=n_fft)
    power_spectrum = np.abs(fft[:n_fft // 2])**2
    power_spectrum /= np.sum(power_spectrum)  # normalize to get a probability distribution

    # Compute entropy
    ps_nonzero = power_spectrum[power_spectrum > 0]
    entropy = -np.sum(ps_nonzero * np.log(ps_nonzero))

    # Normalize by maximum entropy
    max_entropy = np.log(len(power_spectrum))
    forecastability_score = 1 - entropy / max_entropy
    return forecastability_score


def is_seasonal(slice_data, alpha=0.05):
    # Compute autocorrelations and confidence intervals
    acfs, confint = acf(slice_data, nlags=len(slice_data)//2, alpha=alpha, fft=False)
    # Exclude lag 0
    lower = confint[1:, 0]
    upper = confint[1:, 1]
    significant = (lower > 0) | (upper < 0)
    return significant.any()


def compute_seasonality_percent(ts, slice_len=20):
    n_slices = len(ts) // slice_len
    seasonal_count = 0
    for i in range(n_slices):
        slice_data = ts[i*slice_len : (i+1)*slice_len]
        if is_seasonal(slice_data):
            seasonal_count += 1
    return seasonal_count / n_slices


def compute_slice_trend(slice_data):
    t = np.arange(len(slice_data)).reshape(-1, 1)
    y = slice_data.reshape(-1, 1)
    model = LinearRegression().fit(t, y)
    slope = model.coef_[0][0]
    scale = np.mean(np.abs(y)) + 1e-8  # Prevent division by zero
    return slope / scale


def compute_trend_over_series(ts, slice_len=20):
    n_slices = len(ts) // slice_len
    trends = []
    for i in range(n_slices):
        slice_data = ts[i*slice_len : (i+1)*slice_len]
        trend = compute_slice_trend(slice_data)
        trends.append(trend)
    return np.mean(trends)


def scale_timeseries2(timeseries, dim=0):
    scaler = minmax_scaler(feature_range=(-1, 1))
    return scaler.fit_transform(timeseries, dim=dim)


def remove_mean(timeseries, dim: int=0):
    """Removes mean from ``timeseries`` in dimension ``dim``. Dimensions are preserved."""
    return timeseries - np.mean(timeseries, axis=dim, keepdims=True)


def remove_mean2(timeseries, dim: int=0):
    """Removes mean from ``timeseries`` in dimension ``dim``. Dimensions are preserved."""
    return timeseries - torch.mean(timeseries, dim, keepdim=True)


def label_timeseries(timeseries, model):
    """
    Returns a tuple with ``timeseries`` and a label,
    where a True label denotes stationary time series, whereas a False label signifies transient time series.
    """

    timeseries     = torch.unsqueeze(timeseries, 0)
    lookback       = timeseries[:, :model.lookback_nsample]
    timeseries_pre = model.operator_stat(lookback)[0]

    err_fn  = torch.nn.MSELoss(reduction='mean')
    err     = err_fn(timeseries_pre, timeseries)

    timeseries = torch.squeeze(timeseries, 0)
    label      = True if err < model.maxerr_stat else False

    return timeseries, label


def label_stationarity(model, res_max, datadir, timeseries_nsample):
    """
    Based on DMD model residuals, creates a set of labels for stationary/transient data.
    """
    lookback_nsample = model.lookback_nsample
    data             = read_datafile(f'{datadir}/test', timeseries_nsample)
    labels           = torch.zeros(data.shape[0], dtype=torch.bool)

    for this, timeseries in enumerate(data):
        timeseries = torch.unsqueeze(timeseries, 0)

        # --! call stationary operator
        model_i  = timeseries[:, :lookback_nsample]
        model_o  = model.operator_stat(model_i)

        timeseries_pre_mean = model_o[0]

        res_fn  = torch.nn.MSELoss(reduction='mean')
        res     = res_fn(timeseries_pre_mean, timeseries)

        label = True if res < res_max else False
        labels[this] = label

    return labels


def sample_timeseries2(nsample, rng, jdata, *data):

    # --! select current datum
    datum         = data[jdata]
    datum_nsample = len(datum)

    # --! randomly locate a suitable sample region inside the selected datum
    sample_start = int((datum_nsample - nsample) * rng.random())
    sample_end   = sample_start + nsample
    sample       = datum[sample_start:sample_end, :]

    # --! remove the mean of this sample and scale it between -1 and 1
    return scale_timeseries2(remove_mean2(sample))


def sample_timeseries(rng, nsample, timeseries_nsample, jtimeseries, *timeseries_bank):
    """
    Randomly samples timeseries of size ``nsample`` from given ``timeseries_bank``. The array
    ``timeseries_bank`` is indexed using ``jtimeseries``. Timeseries in ``timeseries_bank``
    are expected to be shaped as column vectors [T, 1],
    where T is the number of timesteps.
    """

    # --! select current timeseries
    timeseries = timeseries_bank[jtimeseries]

    # --! randomly locate a suitable sample region inside selected timeseries
    sample_start = int((timeseries_nsample - nsample) * rng.random())
    sample_end   = sample_start + nsample
    sample       = timeseries[sample_start:sample_end, :]

    # --! remove the mean of this sample and scale it between -1 and 1
    sample = scale_timeseries(remove_mean(sample))

    return sample


def next_index(j, n):
    """Gets the next index."""
    return np.remainder(j, n)


def save_mixed_dataset(dir_stat, dir_trans, timeseries_nsample, savedir):
    
    dataconfig = [
        'train1',
        'train2',
        'train3',
        'train4',
        'train5',
        'train6',
        'train7',
        'valid',
        'test'
    ]

    for cfg in dataconfig:
        data_stat  = read_datafile(dir_stat  + '/' + cfg, timeseries_nsample)
        data_trans = read_datafile(dir_trans + '/' + cfg, timeseries_nsample)

        data_stack = torch.stack([data_stat, data_trans], dim=1)
        data_mix   = torch.flatten(data_stack, start_dim=0, end_dim=1)

        mixed_size = data_stat.shape[0]
        data_mix   = data_mix[:mixed_size, :]
        write_datafile(f'{savedir}/{cfg}', data_mix)

def create_dataset(size, model, rng, data):

    timeseries_nsample = model.lookback_nsample + model.forecast_nsample
    ndata              = len(data)

    dataset = [label_timeseries(
        sample_timeseries2(
            timeseries_nsample,
            rng,
            next_index(j, ndata),
            *data), model) for j in range(size)]

    return dataset

def save_trans(model, savedir, data):

    dataconfig = [
        # number of timeseries in a file, file name
        (3500, 'train1'),
        (3500, 'train2'),
        (3500, 'train3'),
        (3500, 'train4'),
        (3500, 'train5'),
        (3500, 'train6'),
        (3500, 'train7'),
        (1000,  'valid'),
        (500,    'test')
    ]

    for this, cfg in enumerate(dataconfig):

        # --! initialize a random number generator with a new seed
        rng = np.random.default_rng(seed=this + 1)

        # --! increase requested data by a factor to make sure there is enough transient data
        size_factor   = 2

        dataset       = create_dataset(cfg[0] * size_factor, model, rng, data)
        dataset_trans = torch.stack([item for item, stat in dataset if not stat], dim=0)

        d1, d2 = torch.split(dataset_trans, [cfg[0], dataset_trans.shape[0] - cfg[0]], dim=0)

        write_datafile(f'{savedir}/{cfg[1]}', d1)

def save_stat(model, savedir, data, size_factor:int=100):

    dataconfig = [
        # number of timeseries in a file, file name
        (3500, 'train1'),
        (3500, 'train2'),
        (3500, 'train3'),
        (3500, 'train4'),
        (3500, 'train5'),
        (3500, 'train6'),
        (3500, 'train7'),
        (1000,  'valid'),
        (500,    'test')
    ]

    for this, cfg in enumerate(dataconfig):

        # --! initialize a random number generator with a new seed
        rng = np.random.default_rng(seed=this + 1)

        dataset       = create_dataset(cfg[0] * size_factor, model, rng, data)
        dataset_stat  = torch.stack([item for item, stat in dataset if stat], dim=0)

        d1, d2 = torch.split(dataset_stat, [cfg[0], dataset_stat.shape[0] - cfg[0]], dim=0)

        write_datafile(f'{savedir}/{cfg[1]}', d1)

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


def save_traindata(timeseries, dirname, snippet_nsample):
    """
    Saves ``timeseries`` to files for training. Input ``timeseries`` is a list of column vectors, ``dirname``
    specifies a directory name where files are saved, and ``snippet_nsample`` is the number
    of samples in a snippet sampled from ``timeseries``.
    """
    datadir = dirname

    dataconfig = [
        # number of timeseries in a file, file name
        (3500, 'train1'),
        (3500, 'train2'),
        (3500, 'train3'),
        (3500, 'train4'),
        (3500, 'train5'),
        (3500, 'train6'),
        (3500, 'train7'),
        (1000,  'valid'),
        (500,    'test')
    ]

    ntimeseries        = len(timeseries)
    timeseries_nsample = len(timeseries[0][:, 0])

    if snippet_nsample < timeseries_nsample:
        for this, cfg in enumerate(dataconfig):

            # --! initialize a random number generator with a new seed
            rng = np.random.default_rng(seed=this + 1)

            data = np.stack([
                sample_timeseries(
                    rng,
                    snippet_nsample,
                    timeseries_nsample,
                    next_index(j, ntimeseries),
                    *timeseries) for j in range(cfg[0])], axis=0)

            write_datafile(f'{datadir}/{cfg[1]}', data)
        print('inf >> training files saved')
    else:
        print('err >> saved data size must be less than input timeseries size!')


def save_testdata(timeseries, dirname, snippet_nsample):
    """
    Saves ``timeseries`` into folder named ``dirname`` for model testing.
    The given ``timeseries`` are split into snippets, sized
    according to ``snippet_nsample``.
    """
    timeseries_nsample = len(timeseries[0][:, 0])

    nsnippet = timeseries_nsample // snippet_nsample
    if nsnippet > 0:
        timeseries_nsample = snippet_nsample * nsnippet

        # --! extract the actual saved timeseries and split them into snippets
        snippet = np.split(timeseries[0][:timeseries_nsample, :], nsnippet, axis=0)

        data = np.expand_dims(
            np.concatenate(
                [snip for snip in snippet], axis=0), 0)

        datadir  = dirname
        filename = 'eval'
        write_datafile(f'{datadir}/{filename}', data)
        print('inf >> evaluation file saved')
    else:
        print('err >> saved timeseries sizes do not match!')


def scale_timeseries(timeseries):
    """Scales ``timeseries`` using from -1 to 1 using min-max algorithm from scikit-learn package.

    The ``timeseries`` are expected to be shaped as [T, N], where T and N
    are the number of timesteps and features, respectively.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    return scaler.fit_transform(timeseries)

