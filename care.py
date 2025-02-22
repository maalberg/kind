import torch
from matplotlib import pyplot as plt

from abc import abstractmethod
from abc import ABCMeta as interface

import utilities as utils

class detuning(torch.nn.Module):
    """
    An autoencoder-based model for a cavity resonance detuning.
    """

    # --! number of dimensions in one eigenfunction, such
    # --! that an n-dimensional eigenfunction represents one oscillator
    # --!
    # --! current eigenfunction format includes two dimensions for the
    # --! displacement and velocity of an oscillator
    efn_dims_n = 2

    # --! number of parameters that are involved in building A and B matrices
    a_params_n = 2
    b_params_n = 1

    u_dims_n   = 1

    def __init__(self, config) -> None:
        super().__init__()

        starts_n  = config['batch_size']
        x_dims_n  = config['x_dims_n']
        u_dims_n  = config['u_dims_n']
        efns_n    = config['modes_n']
        z_dims_n  = efns_n * detuning.efn_dims_n

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([2*i + 1 for i in range(efns_n)]), dim=0), dim=0)
        self._param_b_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(z_dims_n)]), dim=0), dim=0)
        self._z_ic_indices = indices.repeat(starts_n, 1, 1)

        self.cfg = config

        self.enc    = _encoder(x_dims_n + u_dims_n, z_dims_n)
        self.dec    = _decoder(z_dims_n, x_dims_n)
        self.est_as = _estimator_pha(efns_n=efns_n, est_dims_n=detuning.a_params_n)
        self.est_bs = _estimator_amp(efns_n=efns_n, est_dims_n=detuning.b_params_n)

    def fit(self, x, u, pretrain_autoencoder: bool = False, now: bool = False) -> torch.Tensor:

        # --!------------------------------------------------------------------
        # --! initialization

        # --! prepare loss weights
        loss_w_ae     = self.cfg['loss_w_ae']
        loss_w_lin    = self.cfg['loss_w_lin']
        loss_w_pred   = self.cfg['loss_w_pred']
        loss_w_params = self.cfg['loss_w_params']
        loss_w_phys   = self.cfg['loss_w_phys']

        # --!------------------------------------------------------------------
        # --! use an autoencoder to decompose x and u into modes

        # --! according to physics equations, u is squared
        u = torch.square(u)

        # --! decompose x and u into modes and reconstruct back
        z = self.enc(torch.cat([x, u], dim=-1))
        x_ae = self.dec(z)

        # --! fit the loss of decomposition
        loss_ae = loss_w_ae * self._fit_autoencoder(x, x_ae)

        # --! if the autoencoder is to be trained first, then exit here
        if pretrain_autoencoder:
            loss = loss_ae
            return loss, loss_ae, 0., 0., 0.

        # --! then comes a part with a Koopman-based linear prediction
        # --!
        # --! prediction starts from an initial condition (ic) of our latent space
        z_ic = torch.gather(z, -1, self._z_ic_indices[:z.shape[0], :, :])

        a = torch.unsqueeze(self.est_as(z), 1)
        b = torch.unsqueeze(self.est_bs(z), 1)

        # --! having all required data, we predict the trajectory of our latent space z and
        # --! decode it back to the original space
        z_pred  = self._predict_z(z_ic, u, a, b)
        x_pred = self.dec(z_pred)

        # --! fit the losses of linearity and prediction
        loss_lin    = loss_w_lin * self._fit_linearity(z, z_pred)
        loss_pred   = loss_w_pred * self._fit_prediction(x, x_pred)

        #loss_params = loss_w_params * self._fit_params(a)

        loss_phys   = loss_w_phys * self._fit_physics(z, u, a, b, now)

        # --!------------------------------------------------------------------
        # --! output
        # --!

        # --! test ~ test ~ test
        if now:
            with torch.no_grad():

                print(a[0, 0, :])
                print(b[0, 0, :])

                z_dims_n = self.cfg['modes_n'] * detuning.efn_dims_n
                for i in range(z_dims_n):
                    plt.figure()
                    plt.plot(z[0, :, i], label=f'z{i+1}')
                    plt.plot(z_pred[0, :, i], label=f'z{i+1}_pred', linestyle='dashed')
                    plt.legend()
                    plt.show()

                x_dims_n = self.cfg['x_dims_n']
                for i in range(x_dims_n):
                    plt.figure()
                    plt.plot(x[0, :, i], label=f'x{i+1}')
                    plt.plot(x_pred[0, :, i], label=f'x{i+1}_pred', linestyle='dashed')
                    plt.legend()
                    plt.show()

        # --! sum losses together and return the sum
        loss = loss_ae + loss_lin + loss_pred + loss_phys# + loss_params#
        return loss, loss_ae, loss_lin, loss_pred, loss_phys        

    def _fit_autoencoder(self, x, x_ae):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(x_ae, x)

    def _fit_linearity(self, z, z_pred):

        horizon = self.cfg['predict_horizon']
        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(z_pred, z[:, :horizon, :])

    def _fit_prediction(self, x, x_pred):

        horizon = self.cfg['predict_horizon']
        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(x_pred, x[:, :horizon, :])

    def _fit_params(self, params):
        q, w, k = torch.split(params, 1, dim=-1)

        loss_fn = torch.nn.ReLU()
        omega_lo = loss_fn(2*torch.pi*10 - w)
        omega_hi = loss_fn(w - 2*torch.pi*50)
        omega_range = omega_lo + omega_hi

        return omega_range.sum()

    def _fit_physics(self, z, u, a, b, now):

        t_dim = 1
        dt    = self.cfg['timestep']

        z1, z2 = torch.split(z, 1, dim=-1)

        # --! approximate the first and second derivatives of z using finite differences
        dz1 = torch.diff(z1, n=1, dim=t_dim) / dt
        dz2 = torch.diff(z1, n=2, dim=t_dim) / dt**2

        mu, w  = torch.split(a, 1, dim=-1)

        unforced = dz2 + mu * dz1[:, :-1, :] + torch.square(w) * z1[:, :-2, :]
        forced   = b * u[:, :-2, :]
        res      = unforced - forced

        # test
        if now:
            with torch.no_grad():
                plt.figure()
                plt.plot(unforced[0, :, 0], label='unforced')
                plt.legend()
                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.plot(forced[0, :, 0], label='forced')
                plt.legend()
                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.plot(res[0, :, 0], label='residual')
                plt.legend()
                plt.tight_layout()
                plt.show()

        return torch.mean(torch.square(res))

    def predict(self, x, u, horizon):

        # --! actuation u must be squared
        u = torch.square(u)

        z = self.enc(torch.cat([x, u], dim=-1))
        a = torch.unsqueeze(self.est_as(z), 1)
        b = torch.unsqueeze(self.est_bs(z), 1)

        z_ic = torch.gather(z, -1, self._z_ic_indices[:z.shape[0], :, :])

        # --! having all required data, we predict the trajectory of our latent space z and
        # --! decode it back to the original space
        z_pred  = self._predict_z(z_ic, u, a, b)
        x_pred = self.dec(z_pred)

        return x_pred

    def _predict_z(self, z_ic, u, a, b):

        # --!------------------------------------------------------------------
        # --! initialization
        # --!

        horizon = self.cfg['predict_horizon']
        dt      = self.cfg['timestep']

        # --! construct matrices A raised to powers that cover the entire horizon
        mat_a = self._construct_mat_a_pow(a, horizon)

        # u values are also reshaped as [B, H, 1, C_u] to allow broadcasting when multiplying by matrix B
        u = torch.unsqueeze(u, -2)

        # for every horizon position there must be a history of u values multiplied by corresponding ab matrices,
        # so we do the construction of matrices B and a multiplication Bu in one step
        bu = torch.cat([
            torch.sum(
                torch.matmul(
                    u[:, :i, :, :],
                    self._construct_mat_ab(
                        b,
                        mat_a[:, :i, :, :])), 1, keepdim=True) for i in range(1, horizon)], dim=1)

        # moreover, note that z is represented by the initial conditions (ic) of z trajectories
        #
        # furthermore, initial conditions (ic) of z are reshaped as [B, 1, 1, C_z] to allow tensor broadcasting
        # when multiplying by matrices A, which basically means that for every 
        # trajectory we have one initial condition shaped as [1, C_z],
        # where C_z denotes the number of dimensions in z
        z_ic = torch.unsqueeze(z_ic, -2)

        # predict z by multiplying initial conditions of its trajectories by powered matrices A
        #
        # both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, C_efn], i.e.
        # batches with z trajectories consisting of inidividual points [1, C_z]
        #
        # the result is then summed with Bu product
        z_pred = torch.matmul(z_ic, mat_a) + bu

        # remove extra singleton dimensions that were needed for the broadcasting of multiplication
        z_pred = torch.squeeze(z_pred, -2)
        z_ic   = torch.squeeze(z_ic, -2)

        return torch.cat([z_ic, z_pred], dim=1)

    def _construct_mat_a(self, mu, omega):
        dt = self.cfg['timestep']
        return torch.exp(mu*dt) * torch.stack([
            torch.stack([torch.cos(omega*dt), -torch.sin(omega*dt)]),
            torch.stack([torch.sin(omega*dt),  torch.cos(omega*dt)])])

    def _construct_mat_a_diag(self, param):

        # split pairs of mu omega parameters along the last, i.e. channel, dimension
        params = torch.split(param, detuning.a_params_n, dim=-1)

        return torch.block_diag(*[self._construct_mat_a(
            param[0, 0],
            param[0, 1]) for param in params])

    def _construct_mat_a_pow(self, params, horizon):

        # construct a matrix for every provided pair of mu omega parameters
        #
        # constructed matrices are shaped as [B, C_efn, C_efn], where B and C_efn are the number of batches
        # and eigenfunction channels, respectively
        mat = torch.stack([self._construct_mat_a_diag(param[torch.newaxis, 0]) for param in params], dim=0)

        # raise constructed matrices to powers covering all horizon
        #
        # powered matrices are shaped as [B, H - 1, C_efn, C_efn], where H is the number of horizon steps and -1
        # is because we do not predict the first time step
        #
        # note that rotation matrices are transposed to allow multiplication with points
        # shaped as rows, e.g. [1, C_efn]
        return torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(mat, i), -2, -1) for i in range(1, horizon)], dim=1)

    def _construct_mat_ab(self, param_b, mat_a):

        efns_n    = self.cfg['modes_n']
        z_dims_n  = efns_n * detuning.efn_dims_n

        # --! scatter b matrix parameters in a zero-filled matrix
        #
        # --! note that matrix B is constructed as transposed, i.e. shaped as [1, C_z],
        # --! where C_z is the number of channels in a z latent space
        #
        # --! so matrix B is shaped as [B, 1, C_z], where B is the number of batches
        bch_n = param_b.shape[0]
        mat_b = torch.zeros(bch_n,
                            1,
                            z_dims_n,
                            dtype=param_b.dtype).scatter_(-1,
                                                          self._param_b_indices[:bch_n, :, :],
                                                          param_b)

        # --! add an extra singleton dimension after the batch dimension to allow broadcasting
        # --! during multiplication with matrix A
        mat_b = torch.unsqueeze(mat_b, 1)

        if mat_a.shape[1] < 2:
            return mat_b

        # --! multiply matrices A and B
        #
        # --! the idea is that u's (force inputs) that come after the first one must be affected
        # --! by the previous dynamics, i.e. by the previous version of matrix A, so
        # --! corresponding matrices B must be transformed by the right A's
        #
        # --! here, matrices A start with A^1, A^2, etc., and these matrices are transposed (!), so
        # --! we multiply these matrices with transposed matrices B to get B*A, B*A^2 and so on,
        # --! excluding the final matrix A^(H - 1), where H is the horizon of prediction
        #
        # --! finally, matrices BA must be shaped as [B, H - 2, 1, C_z]
        mat_ab = torch.matmul(mat_b, mat_a[:, :-1, :, :])

        # --! return the final version of matrices B by concatenating matrices AB with a plain matrix B
        #
        # --! matrices B must now be shaped as [B, H - 1, 1, C_z]
        #
        # --! matrix B is positioned last to allow a multiplication with u values such that u_(k + h - 1)
        # --! value, i.e. the one before a horizon h, is multiplied with the plain matrix B
        return torch.cat([mat_ab, mat_b], dim=1)

    def _construct_mat_b_diag(self, param):

        # --! split pairs of b parameters along the last, i.e. channel, dimension
        #
        # --! split parameters are shaped as [1, b_params_n * efns_n], where b_params_n
        # --! and efns_n are the number of B matrix parameters and
        # --! eigenfunctions, respectively.
        params = torch.split(param, detuning.b_params_n, dim=-1)

        # --! construct B matrices in a block-diagonal manner
        #
        # --! note that the resulting block-diagonal B matrix is transposed
        return torch.block_diag(*params)

class _encoder(torch.nn.Module):
    def __init__(self, x_dims_n: int=2, z_dims_n: int=2):
        super().__init__()
        self.net = utils.fcnn(features=[x_dims_n, 64, 64, z_dims_n], act_fn_hidden='relu')

    def forward(self, x):
        return self.net(x)


class _decoder(torch.nn.Module):
    def __init__(self, z_dims_n: int=2, x_dims_n: int=2):
        super().__init__()
        self.net = utils.fcnn(features=[z_dims_n, 64, 64, x_dims_n], act_fn_hidden='relu')

    def forward(self, z):
        return self.net(z)


class _estimator(torch.nn.Module, metaclass=interface):
    """
    An internal neural network to help estimate the parameters of system matrices.

    This abstract class follows the template design pattern. Specifically, its ``forward``
    method defines the structure, or template, of an estimation algorithm, so
    that its subclasses must provide the details of this algorithm.
    """
    def __init__(self, efns_n: int=1, efn_dims_n: int=2, est_dims_n: int=1):
        super().__init__()
        self.nets = torch.nn.ModuleList(
            [utils.fcnn(features=[efn_dims_n, 32, 32, est_dims_n], act_fn_hidden='relu') for _ in range(efns_n)])

    def forward(self, z):
        efns = torch.split(z, detuning.efn_dims_n, dim=-1)

        return torch.cat(
            [torch.mean(
                net(self._parameterize(efn)), dim=1) for net, efn in zip(self.nets, efns)], dim=-1)

    @abstractmethod
    def _parameterize(self, z):
        """Parameterizes estimation algorithm. Subclasses must implement this abstract method."""
        raise NotImplementedError


class _estimator_pha(_estimator):
    def __init__(self, efns_n: int=1, est_dims_n: int=1):
        super().__init__(efns_n=efns_n, efn_dims_n=1, est_dims_n=est_dims_n)

    def _parameterize(self, z):
        """Parameterizes estimation with the phase of ``z`` latent space."""
        z1, z2, _ = torch.split(z, [1, 1, detuning.efn_dims_n - 2], dim=-1)
        return torch.atan2(z1, z2)


class _estimator_amp(_estimator):
    def __init__(self, efns_n: int=1, est_dims_n: int=1):
        super().__init__(efns_n=efns_n, efn_dims_n=1, est_dims_n=est_dims_n)

    def _parameterize(self, z):
        """Parameterizes estimation with the amplitude of ``z`` latent space."""
        z1, z2, _ = torch.split(z, [1, 1, detuning.efn_dims_n - 2], dim=-1)
        return torch.square(z1) + torch.square(z2)

