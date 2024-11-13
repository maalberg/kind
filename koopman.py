import torch
import utilities as utils

class eigenfunction:
    def __init__(self, data_dims_n : int = 2, eigenfunc_dims_n : int = 2, inversed : bool = False):
        """
        Constructs a fully-connected neural network that transforms input data into
        Koopman eigenfunctions (or vice versa if ``inversed`` is set to True).
        The underlying neural network is parameterized by the number of
        dimensions in data and eigenfunction, represented by
        parameters ``data_dims_n`` and ``eigenfunc_dims_n`` respectively.
        """

        # define the structure of a fully-connected neural network
        net_features = [data_dims_n, 80, 80, eigenfunc_dims_n]

        # create a fully-connected neural network that will learn the transformation from input
        # data to Koopman eigenfunctions
        self.net = utils.fcnn(
            features=net_features if inversed==False else list(reversed(net_features)),
            activation='relu')

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)

    def parameters(self):
        return self.net.parameters()

class eigenvalue:
    def __init__(self, eigenfunc_dims_n : int = 2) -> None:
        """
        Creates a fully-connected neural network-based operator to transform
        Koopman eigenfunctions into eigenvalues.
        """

        # eigenvalues will be derived from radial eigenfunctions, and these
        # can be described in a two-dimensional space
        self.radial_dims_n = 2

        # since an eigenfunction may have more dimensions than 2, a respective number of
        # neural networks is created to process two-dimensional parts
        # of the eigenfunction space in parallel
        nets_n = int(eigenfunc_dims_n / self.radial_dims_n)
        self.nets = [utils.fcnn(features=[1, 170, 1], activation='relu') for _ in range(nets_n)]

    def __call__(self, eigenfunction : torch.Tensor) -> torch.Tensor:
        """
        Transforms a Koopman ``eigenfunction`` into eigenvalue(s).
        """

        # since there may be more than two dimensions in an eigenfunction, such eigenfunction
        # is split dimension-wise into two-dimensional 'sub-eigenfunctions'
        eigenfuncs_radial = torch.split(eigenfunction, self.radial_dims_n, dim=1)

        # apply a dedicated neural network to each two-dimensional eigenfunction
        #
        # note how a two-dimensional eigenfunction is first constrained to respect radial symmetry
        return torch.cat([
            net(eigenvalue.constrain_radius(eigenfunc)) for net, eigenfunc in zip(self.nets, eigenfuncs_radial)], dim=1)

    def parameters(self):
        params = []
        for net in self.nets: params.extend(list(net.parameters()))
        return params

    @staticmethod
    def constrain_radius(eigenfunc_radial : torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(eigenfunc_radial), dim=1, keepdim=True)
