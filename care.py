import torch

import utilities as utils


# ---------------------------------------------------------------------------*/
# - deep Koopman neural network for a dynamic mode decomposition

class deep_koopman:
    """
    Creates an autoencoder-based neural network that is expected to decompose
    dynamical modes of a system from simulated or experimental data.
    """
    def __init__(self, configuration: dict) -> None:
        """
        Constructs an instance of a dynamic mode decomposition based on the given ``configuration``.
        The ``configuration`` is a dictionary with key-value pairs.
        """
        self.cfg = configuration

        data_dims_n      = self.cfg['data_ch_n']
        eigenfunc_dims_n = len(self.cfg['modes']) * 2

        # based on a typical autoencoder framework, create an encoder that will decompose
        # input data into Koopman eigenfunctions
        self.decomposer = eigenfunction(data_dims_n, eigenfunc_dims_n)

        # as usual for autoencoders, encoded data needs to be decoded back, so create a decoder
        # that will compose/reconstruct data back to its original state
        self.reconstructor = eigenfunction(data_dims_n, eigenfunc_dims_n, inversed=True)

        # create a neural network that will derive dynamics from the decomposed eigenfunctions
        self.dynamics = eigenvalue(eigenfunc_dims_n)

        self._init_mode(configuration)

    def fit(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Fits internal neural networks to the given ``timeseries``. The ``timeseries`` must be formatted
        as [B, T, C], where B, T and C are the number of batch elements, time steps and
        data channels, respectively. The method returns a mean square error loss,
        which is meant to be used by an external optimization loop.
        """

        # decompose timeseries into eigenfunctions
        #
        # eigenfunctions are shaped as [B, T, C_efn], where C_efn denotes the number of
        # channels in eigenfunctions, i.e. in the latent space of this autoencoder
        eigenfuncs = self.decomposer(timeseries)

        horizon = self.cfg['horizon']

        # derive eigenvalues
        #
        # derived eigenvalues are shaped as [B, T, C_eva], where C_eva denotes the number of
        # eigenvalues with their properties
        eigenvalues = self.dynamics(eigenfuncs)

        # based on derived eigenvalues, build a rotation matrix for every trajectory
        #
        # built matrices are shaped as [B, 2, 2]
        dt = self.cfg['timestep']
        matrices = torch.stack([self.dynamics.to_rotation_diag(eva[torch.newaxis, 0] * dt) for eva in eigenvalues], dim=0)

        # raise built matrices to powers covering all horizon
        #
        # powered matrices are shaped as [B, H - 1, 2, 2], where H is the number of horizon steps and -1
        # is because we do not predict the first time step
        #
        # note that the rotation matrices are then transposed to allow multiplication with
        # eigenfunction starting values shaped as rows, i.e. [1, C_efn]
        matpows = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(matrices, h), -2, -1) for h in range(1, horizon)], dim=1)

        # extract the starting values of eigenfunctions
        #
        # these starting values are shaped as [B, 1, 1, C_efn] to allow tensor broadcasting when
        # multiplying by rotation matrices
        efn_start = torch.stack([efn[torch.newaxis, torch.newaxis, 0] for efn in eigenfuncs], dim=0)

        # predict eigenfunctions by multiplying their starting values by powered rotation matrices
        #
        # both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, C_efn], i.e.
        # batches with eigenfunction trajectories consisting of inidividual points [1, C_efn]
        efn_pred = torch.matmul(efn_start, matpows)

        # remove singleton dimensions that were needed for the broadcasting of multiplication
        efn_start = torch.squeeze(efn_start, 1)
        efn_pred  = torch.squeeze(efn_pred, -2)

        eigenfuncs_pred = torch.cat([efn_start, efn_pred], dim=1)

        # reconstruct timeseries
        timeseries_recon = self.reconstructor(eigenfuncs_pred)

        err_mode = self._get_mode_err(eigenvalues)

        # prediction loss
        err_pred = torch.mean(torch.square(eigenfuncs[:, 1:horizon, :] - eigenfuncs_pred[:, 1:horizon, :]))

        # reconstruction loss
        err_recon = torch.mean(torch.square(timeseries[:, :horizon, :] - timeseries_recon))

        # L2 regularization to avoid big weights
        err_big_weights = torch.sum(
            torch.cat(
                [torch.square(param.view(-1)) for param in self.parameters()]))

        # L1 regularization
        err_sparse_weights = torch.sum(
            torch.cat(
                [torch.abs(param.view(-1)) for param in self.parameters()]))

        hp_recon          = self.cfg['loss_hp_recon']
        hp_pred           = self.cfg['loss_hp_pred']
        hp_mode           = self.cfg['loss_hp_mode']
        hp_big_weights    = self.cfg['loss_hp_big_weights']
        hp_sparse_weights = self.cfg['loss_hp_sparse_weights']

        err_recon          = hp_recon * err_recon
        err_pred           = hp_pred * err_pred
        err_mode           = hp_mode * err_mode
        err_big_weights    = hp_big_weights * err_big_weights
        err_sparse_weights = hp_sparse_weights * err_sparse_weights

        loss = err_recon + err_pred + err_mode + err_big_weights + err_sparse_weights

        # return the sum of all losses
        return loss, err_recon, err_pred, err_mode

    def predict(self, timeseries: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Predicts a batch of ``timeseries`` in a linear manner. The method takes the start of every timeseries
        in the given batch and predicts it by a number of steps defined by ``horizon``. Therefore,
        the ``timeseries`` are expected to be formatted as [B, T, C], where B, T and C are
        the number of batch elements, time steps and data channels, respectively.
        """

        # take the starting values of the given timeseries while keeping the batch structure
        timeseries_start = self.start_of(timeseries)

        # decompose starting values into corresponding eigenfunctions
        eigenfuncs = self.decomposer(timeseries_start)

        eigenvalues = self.dynamics(eigenfuncs)

        dt = self.cfg['timestep']
        matrices = torch.stack([self.dynamics.to_rotation_diag(eva[torch.newaxis, 0] * dt) for eva in eigenvalues], dim=0)

        matpows = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(matrices, h), -2, -1) for h in range(1, horizon)], dim=1)

        eigenfuncs_start = torch.unsqueeze(eigenfuncs, 1)
        eigenfuncs_pred = torch.matmul(eigenfuncs_start, matpows)

        eigenfuncs_start = torch.squeeze(eigenfuncs_start, 1)
        eigenfuncs_pred  = torch.squeeze(eigenfuncs_pred, -2)

        eigenfuncs_pred = torch.cat([eigenfuncs_start, eigenfuncs_pred], dim=1)

        # reconstruct a predicted eigenfunction back into timeseries
        return self.reconstructor(eigenfuncs_pred)

    def _init_mode(self, cfg: dict) -> None:
        """
        Initializes dynamic modes for subsequent calculation of mode error during training.
        """

        modes = cfg['modes']
        targets_n = cfg['batch_size']

        # assemble indices to extract frequency properties of eigenvalues
        #
        # since an angular frequency is the second property of an eigenvalue, then
        # the indices of a channel dimension are all odd indices
        #
        # by unsqueezing we establish a batch structure in the assembled indices
        indices = torch.unsqueeze(torch.unsqueeze(
            torch.tensor([2*m + 1 for m in range(len(modes))]), dim=0), dim=0)

        # repeat the batch dimension to comply with the number of actual batches/targets
        self._eva_f_indices = indices.repeat(targets_n, 1, 1)

        # 
        targets = torch.cat([
            torch.randn(targets_n, 1) * modes[m][1] + modes[m][0] for m in range(len(modes))], dim=1)
        self._eva_f_targets = torch.unsqueeze(targets, 1)

    def _get_mode_err(self, eigenvalues: torch.Tensor) -> torch.Tensor:

        # correct the size of indices according to the actual size of input eigenvalues
        indices = self._eva_f_indices[:eigenvalues.shape[0], :, :]

        # use corrected indices to gather eigenvalue frequencies from the last, i.e. channel, dimension
        ch_dim = -1
        evas = torch.gather(eigenvalues, ch_dim, indices)

        # correct the size of targets according to the actual size of input eigenvalues
        targets = self._eva_f_targets[:eigenvalues.shape[0], :, :]

        return torch.mean(torch.square(targets - evas))

    def parameters(self):
        """Returns the parameters of internal neural networks."""
        params = []
        modules = self.decomposer, self.reconstructor, self.dynamics
        for module in modules: params.extend(list(module.parameters()))
        return params

    @staticmethod
    def start_of(timeseries: torch.Tensor) -> torch.Tensor:
        """
        Returns the start of every timeseries inside a batch ``timeseries``. Consequently, ``timeseries``
        are expected to be formatted as [B, T, C], where B, T and C are the number of
        batch elements, time steps and data channels, respectively.
        """
        return torch.stack([datum[torch.newaxis, 0] for datum in timeseries], dim=0)


# ---------------------------------------------------------------------------*/
# - eigenfunction

class eigenfunction:
    def __init__(self, timeseries_dims_n: int = 2, eigenfunc_dims_n: int = 2, inversed: bool = False):
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

        # an eigenvalue has two properties: scaling mu and angular frequency omega
        self.eigenvalue_props_n = 2

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
    
    def to_rotation_diag(self, eigenvalues: torch.Tensor):
        """Uses ``eigenvalues`` to assemble a matrix with 2x2 rotation matrices on its diagonal."""

        # split incoming eigenvalues along the last, i.e. channel, dimension
        evas = torch.split(eigenvalues, self.eigenvalue_props_n, dim=-1)

        return torch.block_diag(*[utils.make_rotation(
            exponent=eva[0, 0],
            angle=eva[0, 1]) for eva in evas])

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


# ---------------------------------------------------------------------------*/
# - control input matrix

class control_input:
    pass

