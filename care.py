import torch

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

        # construct autoencoder that establishes an n-dimensional latent space,
        # where n comes from the given configuration
        self.autoencoder = utils.autoencoder(
            features=[self.config['data_ch_n'], 80, 80, self.config['latent_ch_n']],
            activation=self.config['act'])

        # construct a fully-connected neural network that transforms latent coordinates into eigenvalues
        self.eigenvalues = utils.fcnn([2, 170, 1], activation=self.config['act'])

    def predict_impl(self, coord_first: torch.Tensor, eigens: torch.Tensor) -> torch.Tensor:
        """
        Implements prediction logic that takes the first coordinate ``coord_first``, a vector of
        eigenvalues ``eigens`` and returns a vector of predicted coordinates.
        The data must be formatted as [T, C], where T and C are the
        number of timesteps and channels, respectively.

        The T of both ``eigens`` and the returned prediction will be the same, so ``eigens``
        dictate the length of the prediction. Note that ``eigens`` can be
        represented by just one T-times repeated value.
        """

        # prepare the powers of a rotation matrix
        a = utils.rotation_powers(transposed=True)

        # predict the given coordinates with the help of matrix powers
        #
        # note that matrix multiplication A @ x is performed here in a transposed manner, i.e. xT @ AT
        dt = self.config['timestep']
        coords_pred = torch.cat([torch.matmul(coord_first, a.next(w * dt)) for w in eigens])

        # for now, do not return the last predicted element, as this one predicts into the future,
        # and we need to arrange our data accordingly to be able to check the future
        return torch.cat([coord_first, coords_pred[:-1]])

    def fit(self, timeseries_batch: torch.Tensor) -> torch.Tensor:
        """
        Fits internal neural networks to the given ``timeseries_batch``. The batch must be formatted
        as [B, T, C], where B, T and C are the number of batch elements, time steps and
        data channels, respectively. The method returns a mean square error loss,
        which is meant to be used by an external optimization loop.
        """

        # encode and decode the input batch of timeseries
        coords_batch = self.autoencoder.encoder(timeseries_batch)
        timeseries_o_batch = self.autoencoder.decoder(coords_batch)

        # calculate coding/decoding loss
        timeseries_mseloss = torch.nn.MSELoss()
        timeseries_o_loss = timeseries_mseloss(timeseries_o_batch, timeseries_batch)

        # based on latent coordinates, derive the corresponding eigenvalues
        eigens_batch = self.eigenvalues(coords_batch)

        # with the help of derived eigenvalues predict coordinates in a linear manner
        #
        # stack predicted coordinates along a new dimension 0,
        # thus effectively restoring the structure of the given batch
        coords_o_batch = torch.stack(
            [self.predict_impl(coords[torch.newaxis, 0], eigens) for coords, eigens in zip(coords_batch, eigens_batch)], dim=0)

        # calculate prediction loss
        pred_mseloss = torch.nn.MSELoss()
        pred_o_loss = pred_mseloss(coords_o_batch, coords_batch)

        # return the sum of all losses
        return timeseries_o_loss + pred_o_loss

    def predict(self, timeseries_start: torch.Tensor, prediction_steps_n: int) -> torch.Tensor:
        """
        Based on the starting value of timeseries ``timeseries_start``, predicts a number of steps
        into the future given by ``prediction_steps_n``.
        """
        coord = self.autoencoder.encoder(timeseries_start)

        # transform the first coordinate into an eigenvalue and
        # repeat this eigenvalue to cover all prediction length
        eigens = self.eigenvalues(coord).expand(prediction_steps_n, -1)

        coords_pred = self.predict_impl(coord, eigens)
        return self.autoencoder.decoder(coords_pred)

    def parameters(self):
        """Returns the parameters of the internal neural networks."""
        params = []
        modules = self.autoencoder, self.eigenvalues
        for module in modules: params.extend(list(module.parameters()))
        return params
