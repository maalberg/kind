import torch
from matplotlib import pyplot as plt

import utilities as utils


# --!--------------------------------------------------------------------------
#

class deep_koopman(torch.nn.Module):
    def __init__(self, configuration: dict) -> None:
        super().__init__()

        self._init(configuration)

        z_dims_n  = self.cfg['z_dims_n']
        x_dims_n  = self.cfg['x_dims_n']
        u_dims_n  = self.cfg['u_dims_n']

        self.autoencoder   = utils.autoencoder(x_dims_n + u_dims_n, z_dims_n, y_dims_n=x_dims_n)
        self.a_est         = parameter_estimator(1, param_dims_n=2)
        self.b_est         = parameter_estimator(1, param_dims_n=2)
        self.force_preproc = force_preprocessor()

    def _init(self, configuration: dict) -> None:

        self.cfg = configuration

        modes       = self.cfg['modes']
        modes_n     = len(modes)
        starts_n    = self.cfg['batch_size']

        efn_dims_n  = modes_n * 2

        self.cfg['z_dims_n'] = efn_dims_n

        #indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([2*i+1 for i in range(modes_n)]), dim=0), dim=0)
        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(efn_dims_n)]), dim=0), dim=0)
        self._param_b_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(efn_dims_n)]), dim=0), dim=0)
        self._z_ic_indices = indices.repeat(starts_n, 1, 1)

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
        # --! algorithm

        # --! actuation u is preprocessed first to stress the nonlinear nature
        # --! of u, so the preprocessed u is in fact u^2
        u = self.force_preproc(u)

        # --! encoder and decode given inputs
        # --!
        # --! this encoder is not symmetric, since it encodes x and u, but
        # --! decodes only the x back
        z = self.autoencoder.enc(torch.cat([x, u], dim=-1))
        x_ae = self.autoencoder.dec(z)

        # --! fit the loss of the autoencoder
        loss_ae = loss_w_ae * self._fit_autoencoder(x, x_ae)

        # --! if the autoencoder is to be trained first, then exit here
        if pretrain_autoencoder:
            loss = loss_ae
            return loss, loss_ae, 0., 0., 0.

        # --! then comes a part with a Koopman-based linear prediction
        # --!
        # --! prediction starts from an initial condition (ic) of our latent space
        z_ic = torch.gather(z, -1, self._z_ic_indices[:z.shape[0], :, :])

        # --! calculate the amplitude and the phase of a latent space z
        # --!
        # --! the amplitude is used to parameterize a mapping from z to
        # --! matrix B, whereas the amplitude - to parameterize
        # --! the mapping from z to matrix A
        z1, z2 = torch.split(z, 1, dim=-1)
        amp    = torch.square(z1) + torch.square(z2)
        pha    = torch.atan2(z1, z2)

        a   = torch.unsqueeze(self.a_est(pha), 1)
        b   = torch.unsqueeze(self.b_est(amp), 1)

        # --! having all required data, we predict the trajectory of our latent space z and
        # --! decode it back to the original space
        z_pred  = self._predict_z(z_ic, u, a, b)
        x_pred = self.autoencoder.dec(z_pred)

        # --! fit the losses of linearity and prediction
        loss_lin    = loss_w_lin * self._fit_linearity(z, z_pred)
        loss_pred   = loss_w_pred * self._fit_prediction(x, x_pred)
        #loss_params = loss_w_params * self._fit_params(params)
        #loss_phys  = loss_w_phys * self._fit_physics(z, u, params, now)

        # --!------------------------------------------------------------------
        # --! output
        # --!

        # --! test ~ test ~ test
        if now:
            with torch.no_grad():

                print(a[0, 0, :])
                print(b[0, 0, :])

                plt.figure()
                plt.plot(z[0, :, 0], label='z1')
                plt.plot(z_pred[0, :, 0], label='z1_pred', linestyle='dashed')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(z[0, :, 1], label='z2')
                plt.plot(z_pred[0, :, 1], label='z2_pred', linestyle='dashed')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(x[0, :, 0], label='x1')
                plt.plot(x_pred[0, :, 0], label='x1_pred', linestyle='dashed')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(x[0, :, 1], label='x2')
                plt.plot(x_pred[0, :, 1], label='x2_pred', linestyle='dashed')
                plt.legend()
                plt.show()

        # --! sum losses together and return the sum
        loss = loss_ae + loss_lin + loss_pred# + loss_params# + loss_phys
        return loss, loss_ae, loss_lin, loss_pred, 0.

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

    def _fit_physics(self, z, u, params, now):

        t_dim = 1
        dt    = self.cfg['timestep']

        z1, z2 = torch.split(z, 1, dim=-1)

        # --! approximate the first and second derivatives of z using finite differences
        dz1    = torch.diff(z1, n=1, dim=t_dim) / dt
        dz2    = torch.diff(z1, n=2, dim=t_dim) / dt**2

        q, w, k = torch.split(params, 1, dim=-1)

        unforced = dz2 + q * dz1[:, :-1, :] + torch.square(w) * z1[:, :-2, :]
        forced   = k * u[:, :-2, :]
        res      = unforced + forced

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

        # --! according to the differential equation of a mechanical cavity model,
        # --! input u must be squared
        u = self.force_preproc(u)

        z = self.autoencoder.enc(torch.cat([x, u], dim=-1))

        # --! 
        z1, z2 = torch.split(z, 1, dim=-1)
        amp    = torch.square(z1) + torch.square(z2)
        pha    = torch.atan2(z1, z2)

        a   = torch.unsqueeze(self.a_est(pha), 1)
        b   = torch.unsqueeze(self.b_est(amp), 1)

        z_ic = torch.gather(z, -1, self._z_ic_indices[:z.shape[0], :, :])

        # --! having all required data, we predict the trajectory of our latent space z and
        # --! decode it back to the original space
        z_pred  = self._predict_z(z_ic, u, a, b)
        x_pred = self.autoencoder.dec(z_pred)

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
                    self._construct_mat_b(
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
        #return torch.stack([
            #torch.stack([  torch.tensor(0.) ,   torch.tensor(1.)]),
            #torch.stack([  torch.square(w)  ,   w/q             ])])

    def _construct_mat_a_diag(self, param):

        qw_dims_n =  2
        ch_dim    = -1

        # split pairs of q, w parameters along the last, i.e. channel, dimension
        params = torch.split(param, qw_dims_n, dim=ch_dim)

        return torch.block_diag(*[self._construct_mat_a(
            param[0, 0],
            param[0, 1]) for param in params])

    def _construct_mat_a_pow(self, params, horizon):

        # construct a matrix for every provided pair of q w parameters
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

    def _construct_mat_b(self, param_b, mat_a):

        # scatter b matrix parameters in a zero-filled matrix
        #
        # note that matrix B is constructed as transposed, i.e. shaped as [1, C_z],
        # where C_z is the number of channels in a z latent space
        #
        # next, by zero-filling this B matrix right from the start, we declare that
        # the first column of a 1x2 matrix B is a zero, and the b
        # parameter goes into the second column
        #
        # so matrix B is shaped as [B, 1, C_z], where B is the number of batches
        bch_n = param_b.shape[0]
        mat_b = torch.zeros(bch_n,
                            1,
                            2, # fixme
                            dtype=param_b.dtype).scatter_(-1,
                                                          self._param_b_indices[:bch_n, :, :],
                                                          param_b)

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
        # finally, matrices BA must be shaped as [B, H - 2, 1, C_z]
        mat_ab = torch.matmul(mat_b, mat_a[:, :-1, :, :])

        # return the final version of matrices B by concatenating matrices AB with a plain matrix B
        #
        # matrices B must now be shaped as [B, H - 1, 1, C_z]
        #
        # matrix B is positioned last to allow a multiplication with u values such that u_(k + h - 1)
        # value, i.e. the one before a horizon h, is multiplied with the plain matrix B
        return torch.cat([mat_ab, mat_b], dim=1)


class eigenfunction:
    rad_dims_n = 2

# ---------------------------------------------------------------------------*/
# - eigenvalue

class eigenvalue(torch.nn.Module):
    # an eigenvalue has two properties: scaling mu and angular frequency omega
    eva_props_n = 2

    def __init__(self, efn_dims_n: int = 2) -> None:
        """
        Creates a fully-connected neural network-based operator to transform
        Koopman eigenfunctions into eigenvalues.
        """
        super().__init__()

        self.fourier = utils.rff(features=[2, 128], sigma=5.)

        # since an eigenfunction may have more dimensions than 2, a respective number of
        # neural networks is created to process two-dimensional parts
        # of the eigenfunction space in parallel
        nets_n = int(efn_dims_n / eigenfunction.rad_dims_n)
        self.nets = [utils.fcnn(
            features=[256, 128, 64, self.eva_props_n],
            act_fn_hidden='tanh') for _ in range(nets_n)]

    def forward(self, eigenfunctions: torch.Tensor) -> torch.Tensor:
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
            net(self.fourier(efn)) for net, efn in zip(self.nets, eigenfuncs_rad)], dim=2)

    @staticmethod
    def to_rotation_diag(eigenvalues: torch.Tensor):
        """Uses ``eigenvalues`` to assemble a matrix with 2x2 rotation matrices on its diagonal."""

        # split incoming eigenvalues along the last, i.e. channel, dimension
        evas = torch.split(eigenvalues, eigenvalue.eva_props_n, dim=-1)

        return torch.block_diag(*[utils.make_a(
            eva[0, 0],
            eva[0, 1]) for eva in evas])

    @staticmethod
    def constrain_rad(eigenfunctions: torch.Tensor) -> torch.Tensor:
        """
        Constrains an ``eigenfunction`` by its radius. Input ``eigenfunctions`` are expected
        to be shaped as [B, T, 2], where B and T are the number of batches and
        time steps, respectively. Radius is derived from 2 dimensions.
        """
        return torch.sum(torch.square(eigenfunctions), dim=-1, keepdim=True)


class parameter_estimator(torch.nn.Module):
    def __init__(self, x_dims_n, z_dims_n: int = 32, param_dims_n: int = 1):
        super().__init__()

        #self.net = utils.fcnn(features=[x_dims_n, 32, 32, param_dims_n], act_fn_hidden='relu')

        self.rnn = torch.nn.LSTM(x_dims_n, z_dims_n, batch_first=True)
        self.fc = torch.nn.Linear(z_dims_n, param_dims_n)

    def forward(self, x):
        #return torch.mean(self.net(x), dim=1)
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out


class force_preprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = utils.fcnn(features=[1, 32, 32, 1], act_fn_hidden='relu')

    def forward(self, x):
        return self.net(torch.square(x))

