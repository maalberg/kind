import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler as minmax_scaler


def freeze_module(module):
    """Freezes all parameters in the given ``module``."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreezes all parameters in the given ``module``."""
    for param in module.parameters():
        param.requires_grad = True


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
        np.loadtxt('./data/' + name + '.csv', delimiter=',', dtype=np.float32, ndmin=2))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))


def write_datafile(name: str, data, delim: str = ',') -> None:
    """Writes ``data`` to a file named ``name``. The file is written using a comma-separated-value format."""
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt('./data/' + name + '.csv', filedata, fmt='%.14f', delimiter=delim)


class fcnn(torch.nn.Module):
    """
    A fully-connected neural network.
    """
    def __init__(self, features: list[int]=[1, 16 , 1], act_fn_hidden: str='relu', act_fn_out: str='linear') -> None:
        """
        Constructs a fully-connected neural network with specified ``features`` and ``activation``.

        The ``features`` define the number of neurons in the network layers, e.g. a list
        of integers [1, 16, 1] describes a network that accepts a one-dimensional
        input, the network has one hidden layer with 16 neurons, and the
        network produces a one-dimensional output.

        The ``act_fn_hidden`` is a string name for activation functions, e.g. a string 'relu'
        translates into the torch class torch.nn.ReLU. The network will have one
        single ``act_fn_hidden`` everywhere, except for an output layer -
        this one can be specified using ``act_fn_out``. Currently
        supported activation strings/functions are:
        'relu'    torch.nn.ReLU
        'tanh'    torch.nn.Tanh
        'linear'  torch.nn.Identity
        """
        super().__init__()

        # use a helper constant to define the number of hidden layers
        hidden_n = len(features) - 2

        # assemble a list of activation functions;
        # note that the number of hidden layers is incremented to accomodate the output layer,
        # so as long as we are counting hidden layers, these are set to user-specified
        # activation, but when we reach the output layer, it is set to user-defined string
        activations = [act_fn_hidden if i < hidden_n else act_fn_out for i in range(hidden_n + 1)]

        # construct a neural network;
        # note that the bias of all layers, the output included, is set to true
        self.net = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(i, o, bias=True), self.get_activation_by_name(a)]) for i, o, a in zip(
                    features[:-1], features[1:], activations)])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluates this neural network on ``data``."""
        return self.net(data)

    @staticmethod
    def get_activation_by_name(name: str):
        """
        Maps a string ``name`` that specifyes an activation function to a corresponding torch class object.
        """
        a = torch.nn.Identity
        if name == 'relu':
            a = torch.nn.ReLU
        elif name == 'tanh':
            a = torch.nn.Tanh
        elif name == 'sigmoid':
            a = torch.nn.Sigmoid
        elif name == 'linear':
            a = torch.nn.Identity
        else:
            raise ValueError(f'unknown activation function passed: {name}')
        return a()


def sample_timeseries(rng, sample_sz, timeseries_sz, timeseries_i, *timeseries_array):
    """
    Randomly samples timeseries of size ``sample_sz`` from given ``timeseries_array``. The array
    ``timeseries_array`` is indexed using ``timeseries_i``. Timeseries in ``timeseries_array``
    are expected to be shaped as one-dimensional arrays [T],
    where T is the number of timesteps.
    """

    # --! select current timeseries
    timeseries = timeseries_array[timeseries_i]

    # --! randomly locate a suitable sample region inside selected timeseries
    sample_start = int((timeseries_sz - sample_sz) * rng.random())
    sample_end   = sample_start + sample_sz
    sample       = timeseries[sample_start:sample_end, np.newaxis]

    # --! return a sample with mean removed
    return sample - np.mean(sample, axis=0, keepdims=True)


def next_timeseries_i(file_timeseries_i, timeseries_n):
    """Gets the next index of timeseries that are saved to a file."""
    return np.remainder(file_timeseries_i, timeseries_n)


def save_timeseries_train(timeseries, dir_name, snippet_sz):
    """
    Saves timeseries to files for training. Input ``timeseries`` is a list of column vectors, ``dir_name``
    specifies a directory name where files are saved, and ``snippet_sz`` is the length
    of saved timeseries snippets.
    """
    dataset_dir = 'cavity/' + dir_name

    data_config = [
        # number of timeseries in a file, file name
        (3500, 'train1'),
        (3500, 'train2'),
        (3500, 'train3'),
        (3500, 'train4'),
        (3500, 'train5'),
        (3500, 'train6'),
        (3500, 'train7'),
        (1000, 'valid'),
        (500,  'test')
    ]

    timeseries_n  = len(timeseries)
    timeseries_sz = len(timeseries[0][:, 0])

    if snippet_sz < timeseries_sz:
        for this, cfg in enumerate(data_config):

            # --! initialize a random number generator with a new seed
            rng = np.random.default_rng(seed=this + 1)

            data = np.stack([
                sample_timeseries(
                    rng,
                    snippet_sz,
                    timeseries_sz,
                    next_timeseries_i(file_timeseries_i, timeseries_n),
                    *timeseries) for file_timeseries_i in range(cfg[0])], axis=0)

            write_datafile(f'{dataset_dir}/{cfg[1]}', data)
        print('inf >> training files saved')
    else:
        print('err >> saved data size must be less than input timeseries size!')


def remove_mean(timeseries, dim: int=0):
    """Removes mean from ``timeseries`` in dimension ``dim``. Dimensions are preserved."""
    return timeseries - np.mean(timeseries, axis=dim, keepdims=True)

def save_timeseries_eval(timeseries, dir_name, snippet_sz):
    """
    Saves ``timeseries`` into folder named ``dir_name`` for model evaluation.
    The given ``timeseries`` are split into snippets, sized
    according to ``snippet_sz``.
    """
    timeseries_sz = len(timeseries[0][:, 0])

    snippets_n = timeseries_sz // snippet_sz
    if snippets_n > 0:
        timeseries_sz = snippet_sz * snippets_n

        # --! extract the actual saved timeseries and split them into snippets
        snippets = np.split(timeseries[0][:timeseries_sz, :1], snippets_n, axis=0)

        data = np.expand_dims(np.concatenate([remove_mean(snippet) for snippet in snippets], axis=0), 0)

        dataset_dir = 'cavity/' + dir_name
        filename    = 'eval'
        write_datafile(f'{dataset_dir}/{filename}', data)
        print('inf >> evaluation file saved')
    else:
        print('err >> saved timeseries sizes do not match!')


def scale_timeseries(timeseries):
    """
    Scales ``timeseries`` using min-max algorithm. The min-max range is -1 to 1, as this
    range should suit neural network training.
    """
    scaler = minmax_scaler(feature_range=(-1, 1))

    # --! format scaler input as column vectors
    scaler_inp = np.vstack([timeseries]).T
    return scaler.fit_transform(scaler_inp)
