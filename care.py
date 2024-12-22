import torch

import koopman as koop
import utilities as utils


# ---------------------------------------------------------------------------*/
# - dynamic mode decomposition

class dmd:
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
        self.decomposer = koop.eigenfunction(data_dims_n, eigenfunc_dims_n)

        # as usual for autoencoders, encoded data needs to be decoded back, so create a decoder
        # that will compose/reconstruct data back to its original state
        self.reconstructor = koop.eigenfunction(data_dims_n, eigenfunc_dims_n, inversed=True)

        # create a neural network that will derive dynamics from the decomposed eigenfunctions
        self.dynamics = koop.eigenvalue(eigenfunc_dims_n)

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

        # derive eigenvalues
        #
        # derived eigenvalues are shaped as [B, T, C_eva], where C_eva is equal to C_efn/2,
        # i.e. a pair of latent space channels builds an eigenvalue
        eigenvalues = self.dynamics(eigenfuncs)

        # based on derived eigenvalues, build a rotation matrix for every trajectory
        #
        # built matrices are shaped as [B, 2, 2]
        dt = self.cfg['timestep']
        matrices = torch.stack([utils.rotation_dynamics.linearize(eva[0] * dt) for eva in eigenvalues], dim=0)

        # raise built matrices to powers covering all horizon
        #
        # powered matrices are shaped as [B, H - 1, 2, 2], where H is the number of horizon steps and -1
        # is because we do not predict the first time step
        #
        # note that the rotation matrices are then transposed to allow multiplication with
        # eigenfunction starting values shaped as rows, i.e. [1, C_efn]
        horizon = self.cfg['horizon']
        matpows = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(matrices, h), -2, -1) for h in range(1, horizon)], dim=1)

        # extract the starting values of eigenfunctions
        #
        # these starting values are shaped as [B, 1, 1, C_efn] to allow tensor broadcasting when
        # multiplying by rotation matrices
        eigenfuncs_start = torch.stack([efn[torch.newaxis, torch.newaxis, 0] for efn in eigenfuncs], dim=0)

        # predict eigenfunctions by multiplying their starting values by powered rotation matrices
        #
        # both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, C_efn], i.e.
        # batches with eigenfunction trajectories consisting of inidividual points [1, C_efn]
        eigenfuncs_pred = torch.matmul(eigenfuncs_start, matpows)

        # remove singleton dimensions that were needed for the broadcasting of multiplication
        eigenfuncs_start = torch.squeeze(eigenfuncs_start, 1)
        eigenfuncs_pred  = torch.squeeze(eigenfuncs_pred, -2)

        eigenfuncs_pred = torch.cat([eigenfuncs_start, eigenfuncs_pred], dim=1)

        # reconstruct timeseries
        timeseries_recon = self.reconstructor(eigenfuncs_pred)

        # frequency loss
        freq_target = torch.cat([
            torch.randn(
                eigenvalues.shape[0], 1) * self.cfg['modes'][k][1] + self.cfg['modes'][k][0] for k in range(eigenvalues.shape[-1])], dim=1)
        err_mode = torch.mean(torch.square(freq_target - eigenvalues[:, 0, :]))

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
        matrices = torch.stack([utils.rotation_dynamics.linearize(eva[0] * dt) for eva in eigenvalues], dim=0)

        #horizon = self.cfg['horizon']
        matpows = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(matrices, h), -2, -1) for h in range(1, horizon)], dim=1)

        eigenfuncs_start = torch.unsqueeze(eigenfuncs, 1)
        eigenfuncs_pred = torch.matmul(eigenfuncs_start, matpows)

        eigenfuncs_start = torch.squeeze(eigenfuncs_start, 1)
        eigenfuncs_pred  = torch.squeeze(eigenfuncs_pred, -2)

        eigenfuncs_pred = torch.cat([eigenfuncs_start, eigenfuncs_pred], dim=1)

        # predict an eigenfunction into the future with the help of an eigenvalue
        #dt = self.cfg['timestep']
        #eigenfuncs_pred = torch.stack([
            #self._impl_predict_from(
                #efn[torch.newaxis, 0],
                #horizon,
                #utils.rotation_dynamics.linearize(eva[0] * dt)) for efn, eva in zip(eigenfuncs, eigenvalues)], dim=0)

        # reconstruct a predicted eigenfunction back into timeseries
        return self.reconstructor(eigenfuncs_pred)

    def _impl_predict_from(self, efn_start: torch.Tensor, horizon: int, dyn: torch.Tensor) -> torch.Tensor:
        """
        Predicts an eigenfunction starting from the ``eigenfunc_start`` of the eigenfunction and into the future. The
        number of predicted steps is defined by the length of ``eigenvalue`` vector. Note that
        ``eigenvalue`` can also be filled with the same value, thus promoting the
        invariance of an eigenvalue along a trajectory.

        Accordingly, the ``eigenfunc_start`` of an eigenfunction must be formatted as [1, C], whereas ``eigenvalue`` as
        [T, C]. Here, T and C are the number of time steps and data channels, respectively.
        """

        efn = torch.cat([
            self._impl_predict_next(efn_start, efn_next_i, dyn) for efn_next_i in range(1, horizon)])

        return torch.cat([efn_start, efn])

    def _impl_predict_next(self, efn_start: torch.Tensor, efn_next_i: int, dyn: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                efn_start,
                torch.transpose(torch.linalg.matrix_power(dyn, efn_next_i), 0, 1))

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
