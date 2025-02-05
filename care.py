import torch
from matplotlib import pyplot as plt

import utilities as utils


# ---------------------------------------------------------------------------*/
# - deep Koopman neural network for a dynamic mode decomposition

class deep_koopman(torch.nn.Module):
    """
    Creates an autoencoder-based neural network that is expected to decompose
    dynamical modes of a system from simulated or experimental data.
    """
    def __init__(self, configuration: dict) -> None:
        """
        Constructs an instance of a dynamic mode decomposition based on the given ``configuration``.
        The ``configuration`` is a dictionary with key-value pairs.
        """
        super().__init__()

        self._init(configuration)

        z_dims_n  = self.cfg['z_dims_n']
        x_dims_n  = self.cfg['x_dims_n']
        u_dims_n  = self.cfg['u_dims_n']

        self.autoencoder = utils.autoencoder(x_dims_n + u_dims_n, z_dims_n, y_dims_n=x_dims_n)
        self.estimator   = param_estimator(z_dims_n + u_dims_n, param_dims_n=3)

    def _init(self, configuration: dict) -> None:
        """Initializes the ``configuration`` of this class."""

        self.cfg = configuration

        data_dims_n = self.cfg['x_dims_n']
        ctr_dims_n  = self.cfg['u_dims_n']
        modes       = self.cfg['modes']
        modes_n     = len(modes)
        starts_n    = self.cfg['batch_size']
        targets_n   = self.cfg['batch_size']

        efn_dims_n  = modes_n * 2

        self.cfg['z_dims_n'] = efn_dims_n

        eva_all_dims_n = modes_n * eigenvalue.eva_props_n

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(modes_n)]), dim=0), dim=0)
        self._u_0_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([2*i+1 for i in range(modes_n)]), dim=0), dim=0)
        self._param_k_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(data_dims_n)]), dim=0), dim=0)
        self._ts_start_indices = indices.repeat(starts_n, 1, 1)

        indices = torch.unsqueeze(torch.unsqueeze(torch.tensor([i for i in range(efn_dims_n)]), dim=0), dim=0)
        self._z_ic_indices = indices.repeat(starts_n, 1, 1)

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

    def fit(self, x, dx, u, du, pretrain_autoencoder: bool = False, now: bool = False) -> torch.Tensor:

        # --!------------------------------------------------------------------
        # --! initialization
        # --!

        # --! prepare loss weights
        loss_w_ae       = self.cfg['loss_w_ae']
        loss_w_lin      = self.cfg['loss_w_lin']
        loss_w_pred     = self.cfg['loss_w_pred']
        loss_w_phys_enc = self.cfg['loss_w_phys_enc']
        loss_w_phys_dec = self.cfg['loss_w_phys_dec']

        # --!------------------------------------------------------------------
        # --! algorithm
        # --!

        # --! according to the differential equation of a mechanical cavity model,
        # --! input u must be squared
        u = torch.square(u)

        # --! the input of an autoencoder must contain driving force u
        xu = torch.cat([x, u], dim=-1)

        # --! execute the part of the autoencoder
        #
        # --! the decoder is supposed to decode only the oscillation part of the original input
        z = self.autoencoder.enc(xu)
        x_ae = self.autoencoder.dec(z)

        # --! fit the loss of the autoencoder
        loss_ae = loss_w_ae * self._fit_autoencoder(x, x_ae)

        # --! if only the autoencoder is to be trained first, then exit here
        if pretrain_autoencoder:
            loss = loss_ae
            return loss, loss_ae, 0., 0.

        # --! then comes a part with a Koopman-based linear prediction
        # --!
        # --! prediction starts from an initial condition (ic) of our latent space
        z_ic = torch.gather(z, -1, self._z_ic_indices[:z.shape[0], :, :])

        # --! 
        params = self.estimator(torch.cat([z, u], dim=-1))
        params = torch.unsqueeze(params, 1)

        # --! having all required data, we predict the trajectory of our latent space z and
        # --! decode it back to the original space
        z_pred  = self._predict_z(z_ic, u, params)
        x_pred = self.autoencoder.dec(z_pred)

        # --! fit the losses of linearity and prediction
        loss_lin   = loss_w_lin * self._fit_linearity(z, z_pred)
        loss_pred  = loss_w_pred * self._fit_prediction(x, x_pred)

        # --!------------------------------------------------------------------
        # --! output
        # --!

        # --! test ~ test ~ test
        if now:
            with torch.no_grad():

                print(params[0, 0, :])

                plt.figure()
                plt.plot(z[0, :, 0], label='z1')
                plt.plot(z_pred[0, :, 0], label='z1_pred', linestyle='dashed')
                plt.legend()
                plt.show()

                plt.figure()
                plt.plot(z[0, :, 1], label='z2')
                plt.plot(z_pred[0, :, 1], label='z2', linestyle='dashed')
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
        loss = loss_ae + loss_lin + loss_pred
        return loss, loss_ae, loss_lin, loss_pred

        #loss_phys_enc, loss_phys_dec = self._fit_physics(x, dx, u, du, now)

        # decompose timeseries into eigenfunctions
        #
        # eigenfunctions are shaped as [B, T, C_efn], where C_efn denotes the number of
        # channels in eigenfunctions, i.e. in the latent space of this autoencoder
        #z = self.ae.enc(x)
        #x_ae = self.ae.dec(z)

        #coeffs = torch.unsqueeze(torch.unsqueeze(torch.tensor([self.q, self.w, self.k]), dim=0), dim=0)
        #coeffs = coeffs.repeat(x.shape[0], 1, 1)

        # derive complete trajectories of eigenvalues that are shaped as [B, T, C_eva], where C_eva
        # is the number of channels in an eigenvalue
        #
        # these complete eigenvalue trajectories are further used throughout this method
        #eigenvalues = self.dynamics(eigenfunctions)
        #eigenvalues_start = torch.unsqueeze(torch.unsqueeze(torch.tensor([self.q, self.w]), dim=0), dim=0)
        #eigenvalues_start = eigenvalues_start.repeat(x.shape[0], 1, 1)

        # gather the starting points of eigenfunctions with their corresponding eigenvalues
        #
        # the indices of starting points must be corrected according to the current number of batch elements

        #forced_coeff_start = torch.unsqueeze(torch.unsqueeze(torch.tensor([self.k]), dim=0), dim=0)
        #forced_coeff_start = forced_coeff_start.repeat(x.shape[0], 1, 1)

        #horizon = self.cfg['horizon']
        #eigenfunctions_start = torch.gather(z, -1, self._efn_start_indices[:z.shape[0], :, :])
        #eigenfunctions_pred = self._predict_efn(eigenfunctions_start, horizon,
                                                #eigenvalues_start, u, forced_coeff_start)

        # reconstruct timeseries
        #timeseries_recon = self.ae.dec(eigenfunctions_pred)



        #dzdx = torch.autograd.grad(z, x, torch.ones_like(z), retain_graph=True)[0]
        #dzdt = dzdx * dxdt

        #dz1dt_ode = z[:,:,1]
        #dz2dt_ode = -torch.square(coeffs[:,:,1]) * z[:,:,0] - (coeffs[:,:,1]/coeffs[:,:,0]) * z[:,:,1] - coeffs[:,:,2] * torch.square(coeffs[:,:,1]) * u[:,:,0]

        #dzdt_ode = torch.stack([dz1dt_ode, dz2dt_ode], dim=2)

        #loss_fn_phys = torch.nn.MSELoss(reduction='mean')
        #loss_phys = loss_fn_phys(dzdt, dzdt_ode)



        # linearity loss
        #criterion_lin = torch.nn.MSELoss(reduction='mean')
        #err_lin = criterion_lin(z[:, 1:horizon, :], eigenfunctions_pred[:, 1:horizon, :])

        # reconstruction loss
        #criterion_recon = torch.nn.MSELoss(reduction='mean')
        #err_recon = criterion_recon(x[:, :horizon, :], timeseries_recon)

        # L2 regularization to avoid big weights
        #err_big_weights = torch.sum(
            #torch.cat(
                #[torch.square(param.view(-1)) for param in self.parameters()]))

        # L1 regularization
        #err_sparse_weights = torch.sum(
            #torch.cat(
                #[torch.abs(param.view(-1)) for param in self.parameters()]))

        #loss_phys_enc      = loss_wt_phys_enc * loss_phys_enc 
        #loss_phys_dec      = loss_wt_phys_dec * loss_phys_dec

        # loss_phys_enc + loss_phys_dec

    def predict(self, x, u, horizon):

        # our autoencoder receives a multi-dimensional input that combines state x and control u
        xu = torch.cat([x, u], dim=-1)

        z = self.autoencoder.enc(xu)
        z_pred = self._predict_z(z, u)

        xu_pred = self.autoencoder.dec(z_pred)
        x, u = torch.split(xu_pred, 2, dim=-1)
        return x, u

    def _fit_autoencoder(self, x, x_ae):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(x_ae, x)

    def _fit_physics(self, x, dx, u, du, now):

        # our autoencoder receives a multi-dimensional input that combines state x and control u
        xu = torch.cat([x, u], dim=-1)

        z = self.autoencoder.enc(xu)
        xu_ae = self.autoencoder.dec(z)

        # retrieve parameters q, w and k from z and take the mean of resulting parameter trajectories
        params = torch.mean(self.ode_params(z), dim=1, keepdim=True)

        # split parameters into q, w and k
        param_q, param_w, param_k = torch.split(params, 1, dim=-1)

        # prepare the coefficients of differential equations
        #coeffs = torch.unsqueeze(torch.unsqueeze(torch.tensor([self.q, self.w, self.k]), dim=0), dim=0)
        #coeffs = coeffs.repeat(x.shape[0], 1, 1)
        #q  = coeffs[:, :, 0]
        #w  = coeffs[:, :, 1]
        #k  = coeffs[:, :, 2]

        # prepare the variables of differential equations
        z1, z2 = torch.split(z, 1, dim=-1)
        #z1 = z[:, :, 0]
        #z2 = z[:, :, 1]
        #z3 = z[:, :, 2]
        #u  = u[:, :, 0]

        # compute the differential equations of a mechanical cavity model based on an encoded latent space
        dz1 = z2
        dz2 = -torch.square(param_w) * z1 - (param_w/param_q) * z2 - param_k * torch.square(param_w) * torch.square(u)
        dz  = torch.cat([dz1, dz2], dim=2)

        dz_ae = torch.autograd.grad(z, xu, grad_outputs=torch.ones_like(z), retain_graph=True)[0]

        dzdx_ae = dz_ae[:, :, :2]
        dzdx_ae = dzdx_ae * dx

        dzdu_ae = dz_ae[:, :, -1:]
        dzdu_ae = dzdu_ae * du

        dz_ae = dzdx_ae + dzdu_ae

        # compute a time derivative from a latent space z to the output of our decoder
        dx_ae = torch.autograd.grad(xu_ae, z, grad_outputs=torch.ones_like(xu_ae), retain_graph=True)[0]
        dx_ae = dx_ae[:, :, :2] * dz

        # test
        if now:
            with torch.no_grad():
                plt.figure()
                plt.plot(dz_ae[0, :, 0], label='dz1_ae')
                plt.plot(dz[0, :, 0], label='dz1')
                plt.legend()
                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.plot(dz_ae[0, :, 1], label='dz2_ae')
                plt.plot(dz[0, :, 1], label='dz2')
                plt.legend()
                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.plot(dx_ae[0, :, 0], label='dx1_ae')
                plt.plot(dx[0, :, 0], label='dx1')
                plt.legend()
                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.plot(dx_ae[0, :, 1], label='dx2_ae')
                plt.plot(dx[0, :, 1], label='dx2')
                plt.legend()
                plt.tight_layout()
                plt.show()

                print(f'q is {param_q[0, 0, 0]}')
                print(f'w is {param_w[0, 0, 0]}')
                print(f'k is {param_k[0, 0, 0]}')

        loss_fn_enc = torch.nn.MSELoss(reduction='mean')
        loss_fn_dec = torch.nn.MSELoss(reduction='mean')

        loss_enc = loss_fn_enc(dz_ae, dz)
        loss_dec = loss_fn_dec(dx_ae, dx)

        return loss_enc, loss_dec

    def _fit_linearity(self, z, z_pred):

        horizon = self.cfg['predict_horizon']
        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(z_pred, z[:, :horizon, :])

    def _fit_prediction(self, x, x_pred):

        horizon = self.cfg['predict_horizon']
        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(x_pred, x[:, :horizon, :])

    def _predict_z(self, z_ic, u, params):

        # --!------------------------------------------------------------------
        # --! initialization
        # --!

        horizon = self.cfg['predict_horizon']
        dt      = self.cfg['timestep']

        # --! parameters are split into an eigenvalue and a coupling coefficient,
        # --! and since the eigenvalue comes first and has two values
        # --! the size of the split sections is set to 2
        splitsec_sz = 2
        eva, k  = torch.split(params, splitsec_sz, dim=-1)

        # --! construct matrices A raised to powers that cover the entire horizon
        mat_a = self._construct_mat_a_pow(eva * dt, horizon)

        # u values are also reshaped as [B, H, 1, C_u] to allow broadcasting when multiplying by matrix B
        u = torch.unsqueeze(u, -2)

        # for every horizon position there must be a history of u values multiplied by corresponding ab matrices,
        # so we do the construction of matrices B and a multiplication Bu in one step
        bu = torch.cat([
            torch.sum(
                torch.matmul(
                    u[:, :i, :, :],
                    self._construct_mat_b(
                        k,
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

    def _construct_mat_a(self, q, w):
        return torch.exp(q) * torch.stack([
            torch.stack([torch.cos(w), -torch.sin(w)]),
            torch.stack([torch.sin(w),  torch.cos(w)])])

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
                                                          self._param_k_indices[:bch_n, :, :],
                                                          -param_b)

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


class param_estimator(torch.nn.Module):
    def __init__(self, x_dims_n, z_dims_n: int = 64, param_dims_n: int = 1):
        super().__init__()

        # construct a recurrent and a fully-connected neural networks
        self.rnn = torch.nn.LSTM(x_dims_n, z_dims_n, batch_first=True)
        self.fc = torch.nn.Linear(z_dims_n, param_dims_n)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out

