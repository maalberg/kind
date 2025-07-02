# --!--------------------------------------------------------------!
# --! utilities for neural networks
# --!--------------------------------------------------------------!

import torch


def freeze_module(module):
    """Freezes all parameters in the given ``module``."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreezes all parameters in the given ``module``."""
    for param in module.parameters():
        param.requires_grad = True


class fcnn(torch.nn.Module):
    """
    A fully-connected neural network.
    """
    def __init__(self, feat: list[int]=[1, 16 , 1], actfun_hid: str='relu', actfun_o: str='linear') -> None:
        """
        Constructs a fully-connected neural network with specified features ``feat`` and
        activation functions ``actfun_hid`` and ``actfun_o``.

        The features define the number of neurons in the network layers, e.g. a list
        of integers [1, 16, 1] describes a network that accepts a one-dimensional
        input, the network has one hidden layer with 16 neurons, and the
        network produces a one-dimensional output.

        The ``actfun_hid`` is a string name for activation functions in hidden layers,
        e.g. a string 'relu' translates into the torch class torch.nn.ReLU.
        The network will have the same ``actfun_hid`` in all hidden
        layers. For an output layer the activation function is
        specified using ``actfun_o``.
        
        Currently supported activation strings/functions are:
        'relu'    torch.nn.ReLU
        'tanh'    torch.nn.Tanh
        'sigmoid' torch.nn.Sigmoid
        'linear'  torch.nn.Identity
        """
        super().__init__()

        # use a helper constant to define the number of hidden layers
        nhid = len(feat) - 2

        # assemble a list of activation functions;
        # note that the number of hidden layers is incremented to accomodate the output layer,
        # so as long as we are counting hidden layers, these are set to user-specified
        # activation, but when we reach the output layer, it is set to user-defined string
        actfun = [actfun_hid if i < nhid else actfun_o for i in range(nhid + 1)]

        # construct a neural network;
        # note that the bias of all layers, the output included, is set to true
        self.net = torch.nn.Sequential(*[
            torch.nn.Sequential(*[
                torch.nn.Linear(i, o, bias=True), self._get_actfun(a)]) for i, o, a in zip(
                    feat[:-1], feat[1:], actfun)])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Evaluates this neural network on ``data``."""
        return self.net(data)

    @staticmethod
    def _get_actfun(name: str):
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


def extract_poly_deg(polynomial: str='poly_1'):
    """Extracts a degree from the given ``polynomial`` string."""
    tokens = polynomial.split('_', 1)
    if len(tokens) != 2:
        raise ValueError('bad specification of a polynomial')
    deg = tokens[1]
    if not deg.isdigit():
        raise ValueError('bad specification of a polynomial')
    deg = int(tokens[1])
    if deg == 0:
        raise ValueError('zero degree polynomial is not supported')
    return deg
