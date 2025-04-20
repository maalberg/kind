import numpy as np
import torch


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
    filedata = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    np.savetxt('./data/' + name + '.csv', filedata, fmt='%.14f', delimiter=delim)


class rff:
    def __init__(self, features: list[int] = [1, 64], sigma: float = 1.0) -> None:
        super().__init__()

        x_dims_n = features[0]
        z_dims_n = features[1]

        self.w = torch.nn.Linear(x_dims_n, z_dims_n, bias=False)
        torch.nn.init.normal_(self.w.weight, std=sigma)

    def __call__(self, x):
        z = self.w(x)
        return torch.cat([
            torch.cos(2 * torch.pi * z),
            torch.sin(2 * torch.pi * z)], dim=-1)


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
