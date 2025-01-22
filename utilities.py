import numpy as np
import torch


# ---------------------------------------------------------------------------*/
# - read data from a file

def read_datafile(name: str, datachunk_len) -> torch.Tensor:
    """
    Reads data from a file called ``name`` and formats the data based on ``datachunk_len``,
    i.e. the length of one contiguous chunk of data. The file data are expected to be
    in format [T, C], such that the read data could be formatted as [B, T, C],
    where B, T and C are the number of batches, time steps and data channels,
    repectively.
    """
    data = torch.tensor(
        np.loadtxt('./data/' + name + '.csv', delimiter=',', dtype=np.float32))
    datachunks_n = int(data.shape[0] / datachunk_len)

    # return read data in channels-last format
    return torch.reshape(data, (datachunks_n, datachunk_len, data.shape[1]))


# ---------------------------------------------------------------------------*/
# - write data to a file

def write_datafile(name: str, data) -> None:
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt('./data/' + name + '.csv', filedata, fmt='%.14f', delimiter=',')


# ---------------------------------------------------------------------------*/
# - fully-connected neural network

class fcnn(torch.nn.Module):
    def __init__(self, features: list[int] = [1, 16 , 1], act_fn_hidden: str = 'relu', act_fn_out: str = 'linear') -> None:
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
        elif name == 'linear':
            a = torch.nn.Identity
        else:
            raise ValueError(f'unknown activation function passed: {name}')
        return a()


# ---------------------------------------------------------------------------*/
# autoencoder based on fully-connected neural networks

class autoencoder(torch.nn.Module):
    def __init__(self, x_dims_n: int = 2, z_dims_n: int = 2) -> None:
        super().__init__()

        # define the structure of a fully-connected neural network
        net_features = [x_dims_n, 64, 64, z_dims_n]

        self.enc = fcnn(features=net_features, act_fn_hidden='relu')
        self.dec = fcnn(features=list(reversed(net_features)), act_fn_hidden='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes timeseries ``x`` into a latent space z and then immediately decodes z back to ``x``.
        Input ``x`` is expected to be formatted as [B, T, C], where B, T, and C are
        the number of batches, time steps and data channels, respectively.
        """
        return self.dec(self.enc(x))


# ---------------------------------------------------------------------------*/
# - make a rotation matrix

def make_rotation(exponent: torch.Tensor = torch.tensor(0.), angle: torch.Tensor = torch.tensor(0.)) -> torch.Tensor:
    return torch.exp(exponent) * torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle)]),
        torch.stack([torch.sin(angle),  torch.cos(angle)])])

def make_a(q, w):
    return torch.stack([
        torch.stack([torch.tensor(0.), torch.tensor(1.)]),
        torch.stack([-torch.square(w),  -w/q])])
