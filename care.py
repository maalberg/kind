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
        Construct an instance of dynamic mode decomposition based on the given ``configuration``.
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

        # finally, create an extractor of dynamic modes from Koopman eigenfunctions
        self.dynamics = koop.eigenvalue(eigenfunc_dims_n=self.config['osc_n'] * 2)

    def predict_impl(self, eigenfunction: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Predicts the trajectory of an ``eigenfunction`` into the future.
        The eigenfunctions are supposed to be linear, so the dynamics of a predicted eigenfunction
        is encoded in a matrix. At each prediction step this matrix is parameterized
        with a value from ``eigenvalues``. So ``eigenvalues`` define the
        length of the prediction. In fact, ``eigenvalues`` may contain
        a duplicated version of the same eigenvalue if this
        eigenvalue is not supposed to change along
        the eigenfunction trajectory.

        The data must be formatted as [T, C], where T and C are the number of timesteps and channels, respectively.
        """

        # prepare the powers of a rotation matrix
        a = utils.rotation_powers(blocks_n=self.config['osc_n'], transposed=True)

        # predict the given eigenfunction into the future with the help of matrix powers
        #
        # note that matrix multiplication A @ x is performed here in a transposed manner, i.e. xT @ AT
        dt = self.config['timestep']
        eigenfunc_pred = torch.cat([torch.matmul(eigenfunction, a.next(w * dt)) for w in eigenvalues])

        # for now, do not return the last predicted element, as this one predicts into the future,
        # and we need to arrange our data accordingly to be able to check the future
        return torch.cat([eigenfunction, eigenfunc_pred[:-1]])

    def fit(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Fits internal neural networks to the given ``timeseries``. The ``timeseries`` must be formatted
        as [B, T, C], where B, T and C are the number of batch elements, time steps and
        data channels, respectively. The method returns a mean square error loss,
        which is meant to be used by an external optimization loop.
        """

        # reconstruct timeseries
        eigenfuncs = self.decomposer(timeseries)
        timeseries_recon = self.reconstructor(eigenfuncs)

        # calculate reconstruction loss
        criterion_recon = torch.nn.MSELoss()
        loss_recon = criterion_recon(timeseries_recon, timeseries)

        # extract dynamic modes from eigenfunctions
        eigenvalues = torch.stack([self.dynamics(eigenfunc) for eigenfunc in eigenfuncs], dim=0)

        # with the help of derived eigenvalues predict eigenfunctions in a linear manner
        eigenfuncs_pred = torch.stack(
            [self.predict_impl(
                eigenfunc[torch.newaxis, 0], eigenvalue) for eigenfunc, eigenvalue in zip(eigenfuncs, eigenvalues)], dim=0)

        # calculate prediction loss
        criterion_pred = torch.nn.MSELoss()
        loss_pred = criterion_pred(eigenfuncs_pred, eigenfuncs)

        # L2 weight regularization
        loss_reg = torch.sum(
            torch.cat(
                [torch.square(param.view(-1)) for param in self.parameters()]))

        # return the sum of all losses
        return loss_recon + loss_pred + self.config['loss_reg_l2']*loss_reg

    def predict(self, timeseries: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Based on an initial state of ``timeseries``, predicts a number of steps into the future given
        by ``horizon``. The ``timeseries`` are expected to be formatted
        as [1, C], where C is the number of data channels.
        """

        # first, decompose timeseries into corresponding eigenfunctions
        eigenfuncs = self.decomposer(timeseries)

        # find dynamic modes of decomposed eigenfunctions and
        # duplicate these modes (eigenvalues) to cover all prediction horizon,
        # yes, it is assumed that eigenvalues do not change along a state trajectory
        eigenvalues = self.dynamics(eigenfuncs).expand(horizon, -1)

        # predict an eigenfunction into the future and reconstruct it back into timeseries
        eigenfuncs_pred = self.predict_impl(eigenfuncs, eigenvalues)
        return self.reconstructor(eigenfuncs_pred)

    def parameters(self):
        """Returns the parameters of internal neural networks."""
        params = []
        modules = self.decomposer, self.reconstructor, self.dynamics
        for module in modules: params.extend(list(module.parameters()))
        return params
