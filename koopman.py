import torch
import utilities as utils


# ---------------------------------------------------------------------------*/
# - eigenfunction

class eigenfunction:
    def __init__(self, timeseries_dims_n: int = 2, eigenfunc_dims_n: int = 2, inversed : bool = False):
        """
        Constructs a fully-connected neural network that transforms input timeseries into
        Koopman eigenfunctions (or vice versa if ``inversed`` is set to True).
        The underlying neural network is parameterized by the number of
        dimensions in timeseries and eigenfunction, represented by
        parameters ``timeseries_dims_n`` and
        ``eigenfunc_dims_n``, respectively.
        """

        # define the structure of a fully-connected neural network
        net_features = [timeseries_dims_n, 32, 32, eigenfunc_dims_n]

        # create a fully-connected neural network that will learn the transformation from input
        # data to Koopman eigenfunctions
        self.net = utils.fcnn(
            features=net_features if inversed==False else list(reversed(net_features)),
            actfunc='relu')

    def __call__(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Decomposes ``timeseries`` into eigenfunctions. Input ``timeseries`` are expected
        to be formatted as [B, T, C], where B, T, and C are the number of
        batches, time steps and data channels, respectively.
        """
        return self.net(timeseries)

    def parameters(self):
        """Returns the parameters of internal neural networks."""
        return self.net.parameters()


# ---------------------------------------------------------------------------*/
# - eigenvalue

class eigenvalue:
    def __init__(self, eigenfunc_dims_n: int = 2) -> None:
        """
        Creates a fully-connected neural network-based operator to transform
        Koopman eigenfunctions into eigenvalues.
        """

        # eigenvalues will be derived from radial eigenfunctions, and these
        # can be described in a two-dimensional space
        self.radial_dims_n = 2

        # right now an eigenvalue has only one property: angular frequency omega
        self.eigenvalue_props_n = 1

        # since an eigenfunction may have more dimensions than 2, a respective number of
        # neural networks is created to process two-dimensional parts
        # of the eigenfunction space in parallel
        nets_n = int(eigenfunc_dims_n / self.radial_dims_n)
        self.nets = [utils.fcnn(
            features=[1, 128, self.eigenvalue_props_n],
            actfunc='relu') for _ in range(nets_n)]

    def __call__(self, eigenfunctions: torch.Tensor) -> torch.Tensor:
        """
        Transforms Koopman ``eigenfunctions`` into eigenvalues. Input ``eigenfunctions`` are expected
        to be shaped as [B, T, C], where B, T and C are the number of batches, time steps
        and eigenfunction channels, respectively.
        """

        # split an eigenfunction into two-dimensional radial subfunctions
        eigenfuncs_rad = torch.split(eigenfunctions, self.radial_dims_n, dim=2)

        # apply a dedicated neural network to each two-dimensional eigenfunction
        #
        # note how a two-dimensional eigenfunction is first constrained to respect radial symmetry
        #
        # also note how eigenvalues for each radial eigenfunction are concatenated as columns
        # to the result
        return torch.cat([
            net(self.constrain_rad(efn)) for net, efn in zip(self.nets, eigenfuncs_rad)], dim=2)

    def parameters(self):
        """Returns the parameters of internal neural networks."""
        params = []
        for net in self.nets: params.extend(list(net.parameters()))
        return params

    @staticmethod
    def constrain_rad(eigenfunctions: torch.Tensor) -> torch.Tensor:
        """
        Constrains an ``eigenfunction`` by its radius. Input ``eigenfunctions`` are expected
        to be shaped as [B, T, 2], where B and T are the number of batches and
        time steps, respectively. Radius is derived from 2 dimensions.
        """
        return torch.sum(torch.square(eigenfunctions), dim=-1, keepdim=True)
