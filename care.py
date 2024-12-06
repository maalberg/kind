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
        The configuration is a dictionary with key-value pairs.
        """
        self.config = configuration

        data_dims_n      = self.config['data_ch_n']
        eigenfunc_dims_n = self.config['osc_n'] * 2

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
        eigenfuncs = self.decomposer(timeseries)

        # get eigenvalues from eigenfunctions
        eigenvalues = torch.stack([
            self.dynamics(eigenfunc) for eigenfunc in eigenfuncs], dim=0)

        # predict eigenfunctions
        eigenfuncs_pred = torch.stack(
            [self._impl_predict_from(
                eigenfunc[torch.newaxis, 0], eigenvalue) for eigenfunc, eigenvalue in zip(eigenfuncs, eigenvalues)], dim=0)

        # reconstruct timeseries
        timeseries_recon = self.reconstructor(eigenfuncs_pred)

        # frequency loss
        #freq_target = torch.randn_like(eigenvalues)
        #freq_target[:, :, 0] = freq_target[:, :, 0] * 0.5 + 1.0
        #freq_target[:, :, 1] = freq_target[:, :, 1] * 0.5 + 10.0
        #freq_criterion = torch.nn.MSELoss()
        #loss_freq = freq_criterion(eigenvalues, freq_target)

        # prediction loss
        criterion_pred = torch.nn.MSELoss()
        loss_pred = criterion_pred(eigenfuncs_pred, eigenfuncs)

        # reconstruction loss
        criterion_recon = torch.nn.MSELoss()
        loss_recon = criterion_recon(timeseries_recon, timeseries)

        # L2 regularization to avoid big weights
        loss_small = torch.sum(
            torch.cat(
                [torch.square(param.view(-1)) for param in self.parameters()]))

        # return the sum of all losses
        return 1e-2*loss_recon + loss_pred + 1e-8*loss_small# + 1e-3*loss_freq

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

        # find dynamic modes of decomposed eigenfunctions and
        # duplicate these modes (eigenvalues) to cover all prediction horizon,
        # yes, it is assumed that an eigenvalue do not change along a state trajectory
        #
        # keep the batch structure
        eigenvalues = torch.stack(
            [self.dynamics(eigenfunc).expand(horizon, -1) for eigenfunc in eigenfuncs], dim=0)

        # predict an eigenfunction into the future with the help of an eigenvalue
        eigenfuncs_pred = torch.stack([
            self._impl_predict_from(eigenfunc, eigenvalue) for eigenfunc, eigenvalue in zip(eigenfuncs, eigenvalues)])

        # reconstruct a predicted eigenfunction back into timeseries
        return self.reconstructor(eigenfuncs_pred)

    def predict_from(self, start: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Predicts a timeseries starting from a ``start`` position and into the future for a
        number of time steps specified by ``horizon``. The ``start`` must be formatted
        as [1, C], where C is the number of data channels.
        """

        # sample an eigenfunction at the given initial condition
        eigenfunc = self.decomposer(start)

        # find the dynamic mode of the decomposed eigenfunction and
        # duplicate this mode (eigenvalue) to cover all prediction horizon,
        # yes, it is assumed that an eigenvalue does not change along a state trajectory
        eigenvalue = self.dynamics(eigenfunc).expand(horizon, -1)

        # predict an eigenfunction into the future and reconstruct it back into timeseries
        #
        # note that reconstruction interface expects batches, so we need to insert a
        # singleton dimension at the first dimension by unsqueezing
        return self.reconstructor(torch.unsqueeze(
            self._impl_predict_from(eigenfunc, eigenvalue), dim=0))

    def _impl_predict_from(self, eigenfunc_start: torch.Tensor, eigenvalue: torch.Tensor) -> torch.Tensor:
        """
        Predicts an eigenfunction starting from the ``eigenfunc_start`` of the eigenfunction and into the future. The
        number of predicted steps is defined by the length of ``eigenvalue`` vector. Note that
        ``eigenvalue`` can also be filled with the same value, thus promoting the
        invariance of an eigenvalue along a trajectory.

        Accordingly, the ``eigenfunc_start`` of an eigenfunction must be formatted as [1, C], whereas ``eigenvalue`` as
        [T, C]. Here, T and C are the number of time steps and data channels, respectively.
        """

        # prepare the powers of a rotation matrix
        a = utils.rotation_powers(blocks_n=self.config['osc_n'], transposed=True)

        # predict the given eigenfunction from its starting value into the future with the help of matrix powers
        #
        # note that matrix multiplication A @ x is performed here in a transposed manner, i.e. xT @ AT
        t = self.config['timestep']
        eigenfunc = torch.cat([torch.matmul(eigenfunc_start, a.next(omega * t)) for omega in eigenvalue])

        # for now, do not return the last predicted element, as this one predicts into the future,
        # and we need to arrange our data accordingly to be able to check the future
        return torch.cat([eigenfunc_start, eigenfunc[:-1]])

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
