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


class fourier_feature_map:
    def __init__(self, data_dims_n: int = 1, feature_spec: tuple[float, float] = (0., 1.), feature_dims_n: int = 1) -> None:

        feature_mean, feature_std = feature_spec
        self.map = feature_mean + feature_std * torch.randn(data_dims_n, feature_dims_n)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:

        mapped = torch.stack([
            self._impl_map(timeseries) for timeseries in data], dim=0)

        return mapped

    def _impl_map(self, timeseries: torch.Tensor) -> torch.Tensor:
        x = timeseries @ self.map
        x = 2 * torch.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


# ---------------------------------------------------------------------------*/
# - activation function to prune features

class feature_prune(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.k = 3

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Prunes features in ``data`` to leave only a certain number of features requested by the user.
        The pruned features are set to zero.

        NB! There is no additional activation function, e.g. ReLU or TanH, applied to pruned features.
        """

        features_dim = -1
        features_n = self.k * 2

        # determine the indices of preserved features
        #
        # based on a user-defined number, preserved features start from the beginning of the feature space
        features_i = range(features_n)

        # expand the indices of preserved features according to the dimensions of data
        #
        # indices are expanded to the size of a data portion that is going to be extracted/preserved
        data_i = torch.tensor(features_i).expand_as(data[..., :features_n])

        # extract features from data
        features = torch.gather(data, features_dim, data_i)

        # recreate input data with pruned features set to zero
        pruned = torch.zeros_like(data)
        pruned.scatter_(features_dim, data_i, features)

        return pruned


# ---------------------------------------------------------------------------*/
# - fully-connected neural network

class fcnn(torch.nn.Module):
    def __init__(self, features: list[int] = [1, 16 , 1], actfunc: str = 'relu', actfunc_out: str = 'linear') -> None:
        """
        Constructs a fully-connected neural network with specified ``features`` and ``activation``.

        The ``features`` define the number of neurons in the network layers, e.g. a list
        of integers [1, 16, 1] describes a network that accepts a one-dimensional
        input, the network has one hidden layer with 16 neurons, and the
        network produces a one-dimensional output.

        The ``actfunc`` is a string name for activation functions, e.g. a string 'relu'
        translates into the torch class torch.nn.ReLU. The network will have one
        single ``actfunc`` everywhere, except for the output layer - this
        one can be specified using ``actfunc_out``. Currently
        supported activation strings/functions are:
        'relu'    torch.nn.ReLU
        'tanh'    torch.nn.Tanh
        'linear'  torch.nn.Identity
        'prune'   utilities.feature_prune
        """
        super().__init__()        

        # use a helper constant to define the number of hidden layers
        hidden_n = len(features) - 2

        # assemble a list of activation functions;
        # note that the number of hidden layers is incremented to accomodate the output layer,
        # so as long as we are counting hidden layers, these are set to user-specified
        # activation, but when we reach the output layer, it is set to user-defined string
        activations = [actfunc if i < hidden_n else actfunc_out for i in range(hidden_n + 1)]

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
        elif name == 'prune':
            a = feature_prune
        else:
            raise ValueError(f'unknown activation function passed: {name}')
        return a()


# ---------------------------------------------------------------------------*/
# - fully-connected autoencoder-based neural network

class autoencoder(torch.nn.Module):
    def __init__(self, features: list[int] = [1, 16 , 1], activation: str = 'relu') -> None:
        """
        Constructs a fully-connected autoencoder that is based on provided ``features`` and ``activation``.
        The features is a list of integers that define the number of neurons in the layers
        of an underlying encoder. An underlying decoder will be a symmetrical
        reflection of the encoder. The activation is a string that
        specifies a single activation function which is used
        throughout the autoencoder. The exceptions will be
        the output layers of the encoder and decoder,
        as these use a simple identity function.
        """
        super().__init__()

        # construct both encoder and decoder
        #
        # note that the features of the decoder, i.e. its structure, are a symmetrical reflection of the encoder
        self.encoder = fcnn(features, activation)
        self.decoder = fcnn(list(reversed(features)), activation)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Generates a reconstructed version of the given input ``x``. The shape of ``x``
        is [T, C] or [B, T, C], where B, T, C are the number of
        batches, time steps and data channels, respectively.
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x


# ---------------------------------------------------------------------------*/
# - matrix powers

class matrix_powers:
    def __init__(self, matrix: torch.Tensor, transposed=False) -> None:
        """
        Constructs a utility that iterates over ``matrix`` powers. The ``matrix``
        must be a square matrix. The given ``matrix`` can also be ``transposed``
        by setting the corresponding flag to True.

        The matrix power is initialized to A^0 = 1, or identity matrix. The next power
        requested through method next() will be A^1 = A, and so on.
        """
        self.mat = matrix if transposed == False else torch.transpose(matrix, 0, 1)
        self.pow = torch.eye(matrix.shape[0])

    def next(self) -> torch.Tensor:
        """
        Get the next power of a matrix.
        """
        self.pow = torch.matmul(self.mat, self.pow)
        return self.pow


# ---------------------------------------------------------------------------*/
# - rotation powers

class rotation_powers:
    def __init__(self, blocks_n : int = 1, transposed : bool = False):
        """
        Constructs an iterator over powers of a rotation matrix. The rotation
        matrix is initialized with a ``blocks_n`` number of two-dimensional
        identity matrices placed in a block-diagonal matrix. The iterated
        matrices may also be ``transposed``.
        """
        self.rot = torch.block_diag(*[torch.eye(2) for _ in range(blocks_n)])
        self.transposed = transposed

    def next(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Returns the next power of the rotation matrix. The matrix is parameterized
        by the given ``phi`` angle.
        """
        self.rot = torch.matmul(self.rot, rotation_powers.build_rotation_blocks(phi))
        return self.rot if self.transposed == False else torch.transpose(self.rot, 0, 1)

    @staticmethod
    def build_rotation_blocks(phi: torch.Tensor) -> torch.Tensor:
        return torch.block_diag(*[rotation_powers.build_rotation(a) for a in phi])

    @staticmethod
    def build_rotation(phi: torch.Tensor):
        """
        Returns a rotation matrix that is parameterized by angle ``phi``.
        """

        # fixme: rewrite this piece of code in a more fancier way
        cos = torch.cos(phi)
        sin = torch.sin(phi)
        rot = torch.zeros((2, 2))
        rot[0,0] = cos
        rot[1,1] = cos
        rot[0,1] = -sin
        rot[1,0] = sin
        return rot

