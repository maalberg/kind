# --!--------------------------------------------------------------!
# --! utilities for data operations
# --!--------------------------------------------------------------!

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def remove_mean(timeseries, dim: int=0):
    """Removes mean from ``timeseries`` in dimension ``dim``. Dimensions are preserved."""
    return timeseries - np.mean(timeseries, axis=dim, keepdims=True)


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

