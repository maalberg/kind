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

        self._init(configuration)

        efn_dims_n  = self.cfg['efn_dims_n']
        data_dims_n = self.cfg['data_dims_n']
        ctr_dims_n  = self.cfg['ctr_dims_n']

        # based on a typical autoencoder framework, create an encoder that will decompose
        # input data into Koopman eigenfunctions
        self.decomposer = eigenfunction(data_dims_n, efn_dims_n)

        # as usual for autoencoders, encoded data needs to be decoded back, so create a decoder
        # that will compose/reconstruct data back to its original state
        self.reconstructor = eigenfunction(data_dims_n, efn_dims_n, inversed=True)

        # create a neural network that will derive dynamics from the decomposed eigenfunctions
        self.dynamics = eigenvalue(efn_dims_n)

        self.force_coupler = forced_eigenfunction(force_dims_n=ctr_dims_n, efn_dims_n=efn_dims_n)

    def _init(self, configuration: dict) -> None:
        """Initializes the ``configuration`` of this class."""

        self.cfg = configuration

        data_dims_n = self.cfg['data_dims_n']
        ctr_dims_n  = self.cfg['ctr_dims_n']
        modes       = self.cfg['modes']
        modes_n     = len(modes)
        starts_n    = self.cfg['batch_size']
        targets_n   = self.cfg['batch_size']

        efn_dims_n  = modes_n * 2

        self.cfg['efn_dims_n'] = efn_dims_n

        eva_all_dims_n = modes_n * eigenvalue.eva_props_n

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(modes_n)]), dim=0), dim=0)
        self._ctr_start_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([2*i+1 for i in range(modes_n)]), dim=0), dim=0)
        self._forced_coeff_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(data_dims_n)]), dim=0), dim=0)
        self._ts_start_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(efn_dims_n)]), dim=0), dim=0)
        self._efn_start_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(eva_all_dims_n)]), dim=0), dim=0)
        self._eva_start_indices = indices.repeat(starts_n, 1, 1)

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

    def fit(self, timeseries: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """
        """

        # decompose timeseries into eigenfunctions
        #
        # eigenfunctions are shaped as [B, T, C_efn], where C_efn denotes the number of
        # channels in eigenfunctions, i.e. in the latent space of this autoencoder
        eigenfunctions = self.decomposer(timeseries)

        # derive complete trajectories of eigenvalues that are shaped as [B, T, C_eva], where C_eva
        # is the number of channels in an eigenvalue
        #
        # these complete eigenvalue trajectories are further used throughout this method
        eigenvalues = self.dynamics(eigenfunctions)

        forced_coeff = self.force_coupler(force)

        # gather the starting points of eigenfunctions with their corresponding eigenvalues
        #
        # the indices of starting points must be corrected according to the current number of batch elements
        eigenfunctions_start = torch.gather(eigenfunctions, -1, self._efn_start_indices[:eigenfunctions.shape[0], :, :])
        eigenvalues_start = torch.gather(eigenvalues, -1, self._eva_start_indices[:eigenvalues.shape[0], :, :])
        forced_coeff_start = torch.gather(forced_coeff, -1, self._ctr_start_indices[:forced_coeff.shape[0], :, :])

        horizon = self.cfg['horizon']
        eigenfunctions_pred = self._predict_efn(eigenfunctions_start, horizon,
                                                eigenvalues_start, force, forced_coeff_start)

        # reconstruct timeseries
        timeseries_recon = self.reconstructor(eigenfunctions_pred)

        err_mode = self._get_mode_err(eigenfunctions, eigenvalues)

        # prediction loss
        err_pred = torch.mean(torch.square(eigenfunctions[:, 1:horizon, :] - eigenfunctions_pred[:, 1:horizon, :]))

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

    def predict(self, timeseries: torch.Tensor, force: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        """

        # decompose starting values into corresponding eigenfunctions
        eigenfunctions = self.decomposer(timeseries)
        eigenvalues = self.dynamics(eigenfunctions)
        forced_coeff = self.force_coupler(force)

        # take the starting values of the given timeseries while keeping the batch structure
        start = torch.gather(eigenfunctions, -1, self._efn_start_indices[:eigenfunctions.shape[0], :, :])
        eva = torch.gather(eigenvalues, -1, self._eva_start_indices[:eigenvalues.shape[0], :, :])

        # predict eigenfunctions from start up to horizon
        eigenfunctions_pred = self._predict_efn(start, horizon,
                                                eva, force, forced_coeff)

        # reconstruct a predicted eigenfunction back into timeseries
        return self.reconstructor(eigenfunctions_pred)

    def _build_mat_a(self, eva: torch.Tensor, horizon: int) -> torch.Tensor:

        # build a rotation matrix for every provided eigenvalue
        #
        # built matrices are shaped as [B, C_efn, C_efn], where B and C_efn are the number of batches
        # and eigenfunction channels, respectively
        dt = self.cfg['timestep']
        mat = torch.stack([eigenvalue.to_rotation_diag(eva_i[torch.newaxis, 0] * dt) for eva_i in eva], dim=0)

        # raise built matrices to powers covering all horizon
        #
        # powered matrices are shaped as [B, H - 1, C_efn, C_efn], where H is the number of horizon steps and -1
        # is because we do not predict the first time step
        #
        # note that rotation matrices are transposed to allow multiplication with points
        # shaped as rows, e.g. [1, C_efn]
        return torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(mat, i), -2, -1) for i in range(1, horizon)], dim=1)

    def _build_mat_b(self, forced_coeff: torch.Tensor, mat_a: torch.Tensor) -> torch.Tensor:

        # scatter forced coefficients in a zero-filled matrix
        #
        # note that matrix B is constructed as transposed, i.e. shaped as [1, C_efn],
        # where C_efn is the number of eigenfunction channels
        #
        # next, by zero-filling this B matrix right from the start, we declare that
        # the first column of a 1x2 matrix B is a zero, and the forced
        # coefficient goes into the second column
        #
        # so matrix B is shaped as [B, 1, C_efn], where B is the number of batches
        bch_n = forced_coeff.shape[0]
        mat_b = torch.zeros(bch_n,
                            1,
                            self.cfg['efn_dims_n'],
                            dtype=forced_coeff.dtype).scatter_(-1,
                                                               self._forced_coeff_indices[:bch_n, :, :],
                                                               forced_coeff)

        # add an extra singleton dimension after the batch dimension to allow broadcasting
        # during multiplication with matrix A
        mat_b = torch.unsqueeze(mat_b, 1)

        if mat_a.shape[1] < 2:
            return mat_b

        # multiply matrices A and B
        #
        # the idea is that u's (force inputs) that come after the first one must be affected
        # by the previous dynamics, i.e. by the previous version of matrix A, so
        # corresponding matrices B must be transformed by the right A's
        #
        # here, matrices A start with A^1, A^2, etc., and these matrices are transposed (!), so
        # we multiply these matrices with transposed matrices B to get B*A, B*A^2 and so on,
        # excluding the final matrix A^(H - 1), where H is the horizon of prediction
        #
        # finally, matrices BA must be shaped as [B, H - 2, 1, C_efn]
        mat_ab = torch.matmul(mat_b, mat_a[:, :-1, :, :])

        # return the final version of matrices B by concatenating matrices AB with a plain matrix B
        #
        # matrices B must now be shaped as [B, H - 1, 1, C_efn]
        #
        # matrix B is positioned last to allow a multiplication with u values such that u_(k + h - 1)
        # value, i.e. the one before a horizon h, is multiplied with the plain matrix B
        return torch.cat([mat_ab, mat_b], dim=1)

    def _predict_efn(self,
                     start: torch.Tensor, horizon: int,
                     eva: torch.Tensor, force: torch.Tensor, forced_coeff: torch.Tensor) -> torch.Tensor:
        """
        Predicts eigenfunction from a start point ``efn_start`` up to ``horizon``. The start
        point ``efn_start`` must be shaped as [B, 1, C], where B and C_efn are the
        number of batch elements and eigenfunction channels, respectively.
        The predicted eigenfunction is shaped as [B, horizon, C_efn].
        """

        # starting values of eigenfunctions are reshaped as [B, 1, 1, C_efn] to allow tensor broadcasting
        # when multiplying by rotation matrices, which basically means that for every 
        # trajectory we have one start value shaped as [1, C_efn]
        #
        # also, force input values are reshaped [B, H, 1, C_force]
        x = torch.unsqueeze(start, -2)
        u = torch.unsqueeze(force, -2)

        mat_a = self._build_mat_a(eva, horizon)

        bu = torch.cat([
            torch.sum(
                torch.matmul(
                    u[:, :i, :, :],
                    self._build_mat_b(
                        forced_coeff,
                        mat_a[:, :i, :, :])), 1, keepdim=True) for i in range(1, horizon)], dim=1)

        # predict eigenfunctions by multiplying their starting values by powered rotation matrices
        #
        # both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, C_efn], i.e.
        # batches with eigenfunction trajectories consisting of inidividual points [1, C_efn]
        x_pred = torch.matmul(x, mat_a) + bu

        # remove singleton dimensions that were needed for the broadcasting of multiplication
        x = torch.squeeze(x, -2)
        x_pred  = torch.squeeze(x_pred, -2)

        return torch.cat([x, x_pred], dim=1)

    def _get_mode_err(self, efn: torch.Tensor, eva: torch.Tensor) -> torch.Tensor:

        # correct the size of indices according to the actual size of input eigenvalues
        # and corrected indices to gather eigenvalue frequencies from the last, i.e. channel, dimension
        evas = torch.gather(eva, -1, self._eva_f_indices[:eva.shape[0], :, :])

        # correct the size of targets according to the actual size of input eigenvalues
        targets = self._eva_f_targets[:eva.shape[0], :, :]

        return torch.mean(torch.square(targets - evas))

    def parameters(self):
        """Returns the parameters of internal neural networks."""
        params = []
        modules = self.decomposer, self.reconstructor, self.dynamics, self.force_coupler
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
    # we are working with radial eigenfunctions, and these can be described in
    # a two-dimensional phase space
    rad_dims_n = 2

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
    # an eigenvalue has two properties: scaling mu and angular frequency omega
    eva_props_n = 2

    def __init__(self, efn_dims_n: int = 2) -> None:
        """
        Creates a fully-connected neural network-based operator to transform
        Koopman eigenfunctions into eigenvalues.
        """

        # since an eigenfunction may have more dimensions than 2, a respective number of
        # neural networks is created to process two-dimensional parts
        # of the eigenfunction space in parallel
        nets_n = int(efn_dims_n / eigenfunction.rad_dims_n)
        self.nets = [utils.fcnn(
            features=[1, 128, self.eva_props_n],
            actfunc='relu') for _ in range(nets_n)]

    def __call__(self, eigenfunctions: torch.Tensor) -> torch.Tensor:
        """
        Transforms Koopman ``eigenfunctions`` into eigenvalues. Input ``eigenfunctions`` are expected
        to be shaped as [B, T, C], where B, T and C are the number of batches, time steps
        and eigenfunction channels, respectively.
        """

        # split an eigenfunction into two-dimensional radial subfunctions
        eigenfuncs_rad = torch.split(eigenfunctions, eigenfunction.rad_dims_n, dim=2)

        # apply a dedicated neural network to each two-dimensional eigenfunction
        #
        # note how a two-dimensional eigenfunction is first constrained to respect radial symmetry
        #
        # also note how eigenvalues for each radial eigenfunction are concatenated as columns
        # to the result
        return torch.cat([
            net(self.constrain_rad(efn)) for net, efn in zip(self.nets, eigenfuncs_rad)], dim=2)

    @staticmethod
    def to_rotation_diag(eigenvalues: torch.Tensor):
        """Uses ``eigenvalues`` to assemble a matrix with 2x2 rotation matrices on its diagonal."""

        # split incoming eigenvalues along the last, i.e. channel, dimension
        evas = torch.split(eigenvalues, eigenvalue.eva_props_n, dim=-1)

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
# - 

class forced_eigenfunction:
    def __init__(self, force_dims_n: int = 1, efn_dims_n: int = 2) -> None:

        # there are n neural networks serving each pair of radial eigenfunctions
        nets_n = int(efn_dims_n / eigenfunction.rad_dims_n)

        # construct n fully-connected neural networks
        #
        # note that currently the output of these networks is one-dimensional, i.e.
        # this output applies only to one dimension of a radial eigenfunction,
        # the other one will be simply zeroed; TODO: change this if need be
        self.nets = [utils.fcnn(
            features=[force_dims_n, 128, 1],
            actfunc='relu') for _ in range(nets_n)]

    def __call__(self, force: torch.Tensor) -> torch.Tensor:

        # pass input force through neural networks to get force/coupling coefficients
        #
        # note that the outputs of the networks are concatenated along the last,
        # i.e. channel, dimension
        return torch.cat([net(force) for net in self.nets], dim=-1)

    def parameters(self):
        """Returns the parameters of internal neural networks."""
        params = []
        for net in self.nets: params.extend(list(net.parameters()))
        return params

