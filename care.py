import torch
from matplotlib import pyplot as plt

from abc import abstractmethod
from abc import ABCMeta as interface

import utilities as utils

class detuning(torch.nn.Module):
    """
    An autoencoder-based model for a cavity resonance detuning.
    """

    # --! number of dimensions in one eigenfunction, which describes
    # --! an n-dimensional latent oscillator
    efn_dims_n = 2

    # --! number of dimensions in z latent space that are occupied by actuation u
    zu_dims_n = 1

    # --! number of parameters that are involved in building A and B matrices
    a_params_n = 2
    b_params_n = 1

    def __init__(self, config) -> None:
        super().__init__()

        # --! dynamic modes determine the number of eigenfunctions
        efns_n = len(config['fit_horizon'])

        # --! eigenfunctions compose z latent space, so compute the number of z dimensions
        zx_dims_n = efns_n * detuning.efn_dims_n
        z_dims_n  = zx_dims_n + detuning.zu_dims_n

        # --! compute indices that allow making quick gather and scatter operations in pytorch
        #
        # --! for b parameter - this will be every second column in a 1x2 B matrix (transposed!!)
        #
        # --! for the initial conditions of z oscillator space - these are all first values of z oscillator dimensions
        param_b_i = torch.unsqueeze(torch.unsqueeze(torch.tensor([1]), dim=0), dim=0)
        zx_ic_i = torch.unsqueeze(
            torch.unsqueeze(
                torch.tensor(
                    [i for i in range(zx_dims_n)]), dim=0), dim=0)

        # --! repeat the indices according to a batch size
        bch_sz          = config['batch_size']
        self._param_b_i = param_b_i.repeat(bch_sz, 1, 1)
        self._zx_ic_i   = zx_ic_i.repeat(bch_sz, 1, 1)

        # --! input dimensions
        x_dims_n = config['x_dims_n']
        u_dims_n = config['u_dims_n']

        # --! encoder/decoder pair
        self.enc = _encoder(x_dims_n + u_dims_n, z_dims_n)
        self.dec = _decoder(detuning.efn_dims_n, x_dims_n)

        # --! parameter estimators
        self.est_as = torch.nn.ModuleList([_estimator_pha(est_dims_n=detuning.a_params_n) for _ in range(efns_n)])
        self.est_bs = torch.nn.ModuleList([_estimator_amp(est_dims_n=detuning.b_params_n) for _ in range(efns_n)])

        self.config = config

        # --! statistics of fit
        self.w_fit  = [[] for _ in range(efns_n)]
        self.mu_fit = [[] for _ in range(efns_n)]

    def fit(self, x, u, now: bool = False) -> torch.Tensor:

        # --!------------------------------------------------------------------
        # --! initialization

        # --! prepare loss weights
        loss_w_ae     = self.config['loss_w_ae']
        loss_w_lin    = self.config['loss_w_lin']
        loss_w_pred   = self.config['loss_w_pred']
        loss_w_phys   = self.config['loss_w_phys']

        # --!------------------------------------------------------------------
        # --! encoding/decoding of z latent space

        # --! encode x and u into z latent space
        z = self.enc(torch.cat([x, u], dim=-1))

        efns_n    = len(self.config['fit_horizon'])
        zx_dims_n = efns_n * detuning.efn_dims_n

        # --! split z latent space into the states and actuations of latent oscillators
        zx, zu = torch.split(z, [zx_dims_n, detuning.zu_dims_n], dim=-1)

        efns_sum = torch.cat([
            zx[:, :,  ::2].sum(dim=-1, keepdim=True),
            zx[:, :, 1::2].sum(dim=-1, keepdim=True)], dim=-1)

        x_ae = self.dec(efns_sum)

        # --! fit the loss of encoding/decoding
        loss_ae = loss_w_ae * self._fit_autoencoder(x, x_ae)

        # --!------------------------------------------------------------------
        # --! predicting timeseries x in a linear manner

        # --! prediction starts from an initial condition (ic) of z latent space
        zx_ic = torch.gather(zx, -1, self._zx_ic_i[:zx.shape[0], :, :])

        efns    = torch.split(zx, detuning.efn_dims_n, dim=-1)
        efns_ic = torch.split(zx_ic, detuning.efn_dims_n, dim=-1)
        efns_a  = [torch.unsqueeze(est_a(efn), 1) for est_a, efn in zip(self.est_as, efns)]
        efns_b  = [torch.unsqueeze(est_b(efn), 1) for est_b, efn in zip(self.est_bs, efns)]

        # --! predict trajectories of eigenfunctions up to a prescribed horizon
        horizons = self.config['fit_horizon']
        efns_pred = [self._predict_efn(
            efn_ic, efn_a, efn_b,
            zu,
            horizon) for efn_ic, efn_a, efn_b, horizon in zip(efns_ic, efns_a, efns_b, horizons)]

        loss_lin = torch.sum(
            torch.stack(
                [loss_w_lin * self._fit_linearity(
                    efn, efn_pred, horizon) for efn, efn_pred, horizon in zip(efns, efns_pred, horizons)]))

        xs_pred = [self.dec(efn_pred) for efn_pred in efns_pred]

        horizon_min = torch.min(torch.tensor(self.config['fit_horizon']))
        x_pred = torch.cat([x_pred[:, :horizon_min, :] for x_pred in xs_pred], dim=-1)

        x_pred = torch.cat([
            x_pred[:, :,  ::2].sum(dim=-1, keepdim=True),
            x_pred[:, :, 1::2].sum(dim=-1, keepdim=True)], dim=-1)

        loss_pred = loss_w_pred * self._fit_prediction(x, x_pred, horizon_min)

        # --!------------------------------------------------------------------
        # --! fit physics-informed loss

        loss_phys = torch.sum(
            torch.stack(
                [loss_w_phys * self._fit_physics(
                    efn, efn_a, efn_b,
                    zu,
                    now) for efn, efn_a, efn_b in zip(efns, efns_a, efns_b)]))

        # --!------------------------------------------------------------------
        # --! output

        for i in range (efns_n):
            self.w_fit[i].append(efns_a[i][0, 0, 1].item()) # item method is not differentiable
            self.mu_fit[i].append(efns_a[i][0, 0, 0].item())

        # --! test ~ test ~ test
        if now:
            with torch.no_grad():

                for i in range(efns_n):
                    print(efns_a[i][0, 0, :])
                    print(efns_b[i][0, 0, :])

                for efn, efn_pred in zip(efns, efns_pred):
                    for i in range(detuning.efn_dims_n):
                        plt.figure()
                        plt.plot(efn[0, :, i], label=f'zx{i+1}')
                        plt.plot(efn_pred[0, :, i], label=f'zx{i+1}_pred', linestyle='dashed')
                        plt.legend()
                        plt.show()

                x_dims_n = self.config['x_dims_n']
                for i in range(x_dims_n):
                    plt.figure()
                    plt.plot(x[0, :, i], label=f'x{i+1}')
                    plt.plot(x_pred[0, :, i], label=f'x{i+1}_pred', linestyle='dashed')
                    plt.legend()
                    plt.show()

        # --! sum losses together and return the sum
        loss = loss_ae + loss_lin + loss_pred + loss_phys
        return loss, loss_ae, loss_lin, loss_pred, loss_phys        

    def _fit_autoencoder(self, x, x_ae):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(x_ae, x)

    def _fit_linearity(self, z, z_pred, horizon):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(z_pred, z[:, :horizon, :])

    def _fit_prediction(self, x, x_pred, horizon):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(x_pred, x[:, :horizon, :])

    def _fit_physics(self, z, a, b, u, now):

        t_dim = 1
        dt    = self.config['timestep']

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

        z = self.enc(torch.cat([x, u], dim=-1))

        efns_n    = len(self.config['fit_horizon'])
        zx_dims_n = efns_n * detuning.efn_dims_n

        # --! split z latent space into oscillator states and actuation
        zx, zu = torch.split(z, [zx_dims_n, detuning.zu_dims_n], dim=-1)
        zx_ic = torch.gather(zx, -1, self._zx_ic_i[:z.shape[0], :, :])

        efns    = torch.split(zx, detuning.efn_dims_n, dim=-1)
        efns_ic = torch.split(zx_ic, detuning.efn_dims_n, dim=-1)
        efns_a  = [torch.unsqueeze(est_a(efn), 1) for est_a, efn in zip(self.est_as, efns)]
        efns_b  = [torch.unsqueeze(est_b(efn), 1) for est_b, efn in zip(self.est_bs, efns)]

        # --! predict trajectories of eigenfunctions up to a prescribed horizon
        efns_pred = [self._predict_efn(
            efn_ic, efn_a, efn_b,
            zu,
            horizon) for efn_ic, efn_a, efn_b in zip(efns_ic, efns_a, efns_b)]

        x_pred = torch.cat([self.dec(efn_pred) for efn_pred in efns_pred], dim=-1)

        x_pred = torch.cat([
            x_pred[:, :,  ::2].sum(dim=-1, keepdim=True),
            x_pred[:, :, 1::2].sum(dim=-1, keepdim=True)], dim=-1)

        return x_pred

    def _predict_efn(self, z_ic, a, b, u, horizon):

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
        dt = self.config['timestep']
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

        # --! scatter b matrix parameters in a zero-filled matrix
        #
        # --! note that matrix B is constructed as transposed, i.e. shaped as [1, C_efn],
        # --! where C_efn is the number of channels in an eigenfunction
        #
        # --! so matrix B is shaped as [B, 1, C_efn], where B is the number of batches
        bch_n = param_b.shape[0]
        mat_b = torch.zeros(bch_n,
                            1,
                            detuning.efn_dims_n,
                            dtype=param_b.dtype).scatter_(-1,
                                                          self._param_b_i[:bch_n, :, :],
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
        self.net = utils.fcnn(features=[x_dims_n, 64, z_dims_n], act_fn_hidden='linear')

    def forward(self, x):
        return self.net(x)


class _decoder(torch.nn.Module):
    def __init__(self, z_dims_n: int=2, x_dims_n: int=2):
        super().__init__()
        self.net = utils.fcnn(features=[z_dims_n, 64, x_dims_n], act_fn_hidden='linear')

    def forward(self, z):
        return self.net(z)


class _estimator(torch.nn.Module, metaclass=interface):
    """
    An internal neural network to help estimate the parameters of system matrices.

    This abstract class follows the template design pattern. Specifically, its ``forward``
    method defines the structure, or template, of an estimation algorithm, so
    that its subclasses must provide the details of this algorithm.
    """
    def __init__(self, efn_dims_n: int=2, est_dims_n: int=1):
        super().__init__()
        self.net = utils.fcnn(features=[efn_dims_n, 32, est_dims_n], act_fn_hidden='relu')

    def forward(self, efn):
        time_dim = 1
        return torch.mean(self.net(self._parameterize(efn)), dim=time_dim)

    @abstractmethod
    def _parameterize(self, efn):
        """Parameterizes estimation algorithm. Subclasses must implement this abstract method."""
        raise NotImplementedError


class _estimator_pha(_estimator):
    def __init__(self, est_dims_n: int=1):
        super().__init__(efn_dims_n=1, est_dims_n=est_dims_n)

    def _parameterize(self, efn):
        z1, z2 = torch.split(efn, 1, dim=-1)
        return torch.atan2(z1, z2)


class _estimator_amp(_estimator):
    def __init__(self, est_dims_n: int=1):
        super().__init__(efn_dims_n=1, est_dims_n=est_dims_n)

    def _parameterize(self, efn):
        z1, z2 = torch.split(efn, 1, dim=-1)
        return torch.square(z1) + torch.square(z2)


class detune(torch.nn.Module):

    # --! number of nonlinear functions embeddings
    funs_n = 3

    # --! number of parameters for every nonlinear function embedding
    sin_params_n = 2 # amplitude and angular frequency
    cos_params_n = 2
    exp_params_n = 1 # power

    # --! size of dynamic parameter filters encoded by an encoder from timeseries data
    #
    # --! Timeseries are partioned into slices that are encoded by an encoder to
    # --! produce, so to say, dynamic kernels, or filters. These kernels
    # --! help extract specific features, i.e. nonlinear function
    # --! parameters from the raw timeseries. In constrast to
    # --! static kernels in convolutional neural networks,
    # --! the kernels here are dynamic, because they
    # --! are produced from the timeseries every time.
    fun_params_kern_sz = 51

    def __init__(self):
        super().__init__()

        # --! test
        timeseries_dims_n = 1 # displacement
        timeseries_sz     = 102

        self.feats_n      = timeseries_dims_n
        self.timestep     = 0.001

        # --! as the input to an encoder we provide flattened timeseries sliced according to filter sizes
        fun_params_enc_inps_n = detune.fun_params_kern_sz * timeseries_dims_n

        # --! 
        fun_params_enc_outs_n = (
            detune.sin_params_n + detune.cos_params_n + detune.exp_params_n) * detune.fun_params_kern_sz * timeseries_dims_n

        # --! instantiate a multi-layer perceptron as an encoder for function parameter kernels
        self.fun_params_kern_enc = utils.fcnn(
            features=[fun_params_enc_inps_n, 64, 64, fun_params_enc_outs_n],
            act_fn_hidden='relu')

        # --! number of inputs (features) for a function dynamics encoder
        #
        # --! as features this encoder gets timeseries slices, so the idea is to pass
        # --! an n-dimensional trace of function embeddings to the encoder
        #
        # --! note that operation //, i.e. divide and floor, produces an int rather than a float,
        # --! which is what the class constructor expects - an int
        funs_dyn_enc_inps_n = timeseries_sz // detune.fun_params_kern_sz

        # --! a transformer is responsible for deriving the dynamics of timeseries that
        # --! are local to slices
        #
        # --! the transformer attends over nonlinear function embeddings
        self.funs_dyn_enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=funs_dyn_enc_inps_n,
                nhead=1,
                dim_feedforward=128,
                batch_first=True),
            num_layers=3,
            enable_nested_tensor=False)

        # --! this is a self-attention module that is supposed to produce a square matrix
        # --! that describes evolution of function values
        self.funs_dyn = torch.nn.MultiheadAttention(embed_dim=funs_dyn_enc_inps_n, num_heads=1, batch_first=True)

        # --! 
        dec_inps_n = detune.funs_n * timeseries_dims_n
        dec_outs_n = detune.fun_params_kern_sz * timeseries_dims_n

        # --! 
        self.dec = utils.fcnn(features=[dec_inps_n, 64, 64, dec_outs_n], act_fn_hidden='relu')

        # --! training statictics
        self.sin_freq = []
        self.cos_freq = []
        self.exp_power = []

    def forward(self, timeseries):

        timesteps_dim = 1
        timesteps_n   = timeseries.shape[timesteps_dim]

        if timesteps_n % detune.fun_params_kern_sz:
            print('err >> the size of input timeseries is not a multiple of a filter size')
            return

        # --! reshape input timeseries to work with dynamic kernels
        #
        # --! the length of timeseries is reshaped to extract the kernels
        # --! into the prelast dimension, i.e.
        # --! [B, T, C] -> [B, T / kern_sz, T_kern, C], where
        # --! B, T and C are the number of batch elements,
        # --! timesteps and data channels, respectively.
        #
        # --! note that -1 below infers the size of a dimension
        inps = timeseries.reshape(timeseries.shape[0], -1, detune.fun_params_kern_sz, self.feats_n)
        inps = inps.reshape(timeseries.shape[0], -1, detune.fun_params_kern_sz * self.feats_n)

        # --! compute nonlinear function embeddings for all chunks
        #
        # --! the output embeddings are shaped as [B, T / kern_sz, funs_n]
        funs = self._embed_functions(inps)

        # --! predict these embeddings
        funs_pred = self._predict(funs)

        # --! reconstruct function embeddings back to timeseries
        timeseries_recon = self.dec(funs)
        timeseries_pred  = self.dec(funs_pred)

        # --! since timeseries are reconstructed as rows their shape is [B, C, T],
        # --! but the input timeseries are shaped as columns,
        # --! so reshape the reconstructed ones
        timeseries_recon = timeseries_recon.reshape(timeseries_recon.shape[0], -1, self.feats_n)
        timeseries_pred  = timeseries_pred.reshape(timeseries_recon.shape[0], -1, self.feats_n)

        return timeseries_recon, timeseries_pred

    def _embed_functions(self, inps):

        # --! based on inputs, encode parameter kernels (filters)
        kerns = self.fun_params_kern_enc(inps)

        # --! reshape encoded kernels to format their multiplication with inputs, e.g.
        # --!
        # --! kernels: [B, T / kern_sz, fun_params_n * kern_sz * C] ->
        # --! [B, T / kern_sz, fun_params_n, kern_sz, C]
        # --!
        # --! inputs: [B, T / kern_sz, kern_sz * C] ->
        # --! [B, T / kern_sz, kern_sz, C]
        kerns = kerns.reshape(
            inps.shape[0], inps.shape[1],
            detune.sin_params_n + detune.cos_params_n + detune.exp_params_n,
            detune.fun_params_kern_sz * self.feats_n)
        kerns = kerns.reshape(
            inps.shape[0], inps.shape[1],
            detune.sin_params_n + detune.cos_params_n + detune.exp_params_n,
            detune.fun_params_kern_sz, self.feats_n)
        inps = inps.reshape(inps.shape[0], inps.shape[1], detune.fun_params_kern_sz, self.feats_n)

        # --! with the help of kernels extract function parameters from input timeseries
        fun_params = torch.einsum("blkdf, bldf -> blfk", kerns, inps)

        # --! split the parameters of different measurement functions
        sin_params, cos_params, exp_params = torch.split(
            fun_params,
            [
                detune.sin_params_n, detune.cos_params_n,
                detune.exp_params_n],
            dim=-1)

        with torch.no_grad():
            _, sin_freq = torch.split(sin_params, 1, dim=-1)
            self.sin_freq.append(torch.mean(sin_freq))
            _, cos_freq = torch.split(cos_params, 1, dim=-1)
            self.cos_freq.append(torch.mean(cos_freq))
            self.exp_power.append(torch.mean(exp_params))

        # --! measure nonlinear functions, or so-called embeddings, at the current slice of timeseries
        #
        # --! note that there is one single measurement of each function to describe
        # --! a slice, so the granularity of the slicing may play a role
        funs = torch.cat([
            self._meas_sin(sin_params),
            self._meas_cos(cos_params),
            self._meas_exp(exp_params)], dim=-1)

        # --! reshape dimensions to go from a shape [B, 1, C, funs_n]
        # --! to [B, 1, C * func_n]
        #
        # --! note that 1 denotes the currently iterated slice
        return funs.reshape(funs.shape[0], funs.shape[1], -1)

    def _predict(self, funs):

        # --! encode function embeddings to learn which embedding, e.g. a sine,
        # --! describes the current timeseries best, this embedding
        # --! receives then the most attention
        #
        # --! input functions have a shape [B, T / kern_sz, funs_n], however
        # --! the idea is to attend over functions, so we transpose
        # --! the shape to [B, funs_n, T / kern_sz], such that
        # --! function values represent the sequence,
        # --! and timeseries slices - the features
        funs_dyn = self.funs_dyn_enc(funs.transpose(1, 2))

        # --! perform self-attention by passing the above encoder output as
        # --! the query, key and value of an extra attention module
        #
        # --! this produces a square matrix [funs_n, funs_n] that can be
        # --! used for prediction as a system matrix A
        _, funs_dyn_mat = self.funs_dyn(funs_dyn, funs_dyn, funs_dyn)

        # --! we take the number of timeseries slices as a prediction horizon
        horizon = funs.shape[1]

        # --! raise attention matrices to powers covering all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, funs_n, funs_n], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that dynamics matrices are also transposed to allow multiplication with functions
        # --! shaped as rows, i.e. [1, funs_n]
        funs_dyn_mat = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(funs_dyn_mat, i), -2, -1) for i in range(1, horizon)], dim=1)

        # --! extract the initial conditions of function time (per slice) trajectories
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, funs_n] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        funs_ic = torch.unsqueeze(funs[:, :1], -2)

        # --! predict the time (per slice) evolution of functions by multiplying the initial conditions of
        # --! their trajectories by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, funs_n], i.e.
        # --! batches with functions trajectories consisting of inidividual function-points [1, funs_n]
        funs_pred = torch.matmul(funs_ic, funs_dyn_mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        funs_pred = torch.squeeze(funs_pred, -2)
        funs_ic   = torch.squeeze(funs_ic, -2)

        return torch.cat([funs_ic, funs_pred], dim=1)

    def fit(self, timeseries):

        # --! forward input timeseries to the main algorithm
        timeseries_recon, timeseries_pred = self.forward(timeseries)

        loss_ae = self._fit_autoencoder(timeseries, timeseries_recon)
        loss_pred = self._fit_prediction(timeseries, timeseries_pred)

        loss = loss_ae + loss_pred

        return loss, loss_ae, loss_pred

    def _meas_sin(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.sin(self.timestep * freq)

    def _meas_cos(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.cos(self.timestep * freq)

    def _meas_exp(self, params):
        power = params
        return torch.exp(self.timestep * power)

    def _fit_autoencoder(self, timeseries, timeseries_recon):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(timeseries_recon, timeseries)

    def _fit_prediction(self, timeseries, timeseries_pred):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(timeseries_pred, timeseries)

