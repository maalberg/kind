import torch
from matplotlib import pyplot as plt

from abc import abstractmethod
from abc import ABCMeta as interface

from dataclasses import dataclass

import utilities as utils


@dataclass
class config:
    """
    This data class contains configuration for the detuning model.
    """

    timeseries_dims_n: int
    timeseries_sz: int
    timeseries_timestep: float

    # --! basis functions used to build lifted embedding
    #
    # --! this dictionary is structured as
    # --! * function name: str
    # --! * number of function parameters: int
    funs: dict

    # --! size of dynamic parameter filters encoded by an encoder from timeseries data
    #
    # --! Timeseries are partioned into slices that are encoded by an encoder to
    # --! produce, so to say, dynamic kernels, or filters. These kernels
    # --! help extract specific features, i.e. nonlinear function
    # --! parameters from the raw timeseries. In constrast to
    # --! static kernels in convolutional neural networks,
    # --! the kernels here are dynamic, because they
    # --! are produced from the timeseries every time.
    fun_params_kern_sz: int

    fit_weight_lin_global: float
    fit_weight_lin_local: float


class detune(torch.nn.Module):
    """
    This class models detuning of a cavity resonance (care).
    """

    def __init__(self, config):
        super().__init__()

        self.timeseries_dims_n     = config.timeseries_dims_n
        self.timeseries_sz         = config.timeseries_sz
        self.timeseries_timestep   = config.timeseries_timestep

        self.funs                  = config.funs
        self.fun_params_kern_sz    = config.fun_params_kern_sz

        self.fit_weight_lin_global = config.fit_weight_lin_global
        self.fit_weight_lin_local  = config.fit_weight_lin_local

        # --! details of basis functions
        funs_n        = len(self.funs)
        funs_params_n = sum(self.funs.values())

        # --! as the input to a function parameter encoder we provide flattened timeseries
        # --! sliced according to filter/kernel sizes
        fun_params_enc_inps_n = self.fun_params_kern_sz * self.timeseries_dims_n

        # --! 
        fun_params_enc_outs_n = funs_params_n * self.fun_params_kern_sz * self.timeseries_dims_n

        # --! instantiate a multi-layer perceptron as an encoder for function parameter kernels
        self.fun_params_kern_enc_g = utils.fcnn(
            features=[fun_params_enc_inps_n, 64, 64, fun_params_enc_outs_n],
            act_fn_hidden='relu')

        # --! instantiate a multi-layer perceptron as an encoder for function parameter kernels
        self.fun_params_kern_enc_l = utils.fcnn(
            features=[fun_params_enc_inps_n, 64, 64, fun_params_enc_outs_n],
            act_fn_hidden='relu')

        # --! number of inputs (features) for an adaptive function dynamics encoder,
        # --! which acts locally on a slice level
        #
        # --! as features this encoder gets timeseries slices, so the idea is to pass
        # --! an n-dimensional trace of function embeddings to the encoder
        #
        # --! note that operation //, i.e. divide and floor, produces an int rather than a float,
        # --! which is what the class constructor expects - an int
        funs_dyn_enc_inps_n = self.timeseries_sz // self.fun_params_kern_sz

        # --! a transformer is responsible for adaptively deriving the dynamics of timeseries
        # --! that are local to slices
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
        # --! that describes the evolution of local function values
        self.funs_dyn = torch.nn.MultiheadAttention(embed_dim=funs_dyn_enc_inps_n, num_heads=1, batch_first=True)

        # --! a learnable matrix which is supposed to determine global dynamics that
        # --! are shared across slices
        #
        # --! these dynamics are not adaptable - once they are trained, they are fixed,
        # --! so ideally they should capture the general behavior of a system
        timeseries_dyn_inps_n = funs_n * self.timeseries_dims_n
        timeseries_dyn_outs_n = funs_n * self.timeseries_dims_n
        self.timeseries_dyn = torch.nn.Linear(timeseries_dyn_inps_n, timeseries_dyn_outs_n, bias=False)

        # --! 
        dec_inps_n = funs_n * self.timeseries_dims_n
        dec_outs_n = self.fun_params_kern_sz * self.timeseries_dims_n

        # --! 
        self.dec_g = utils.fcnn(features=[dec_inps_n, 64, 64, dec_outs_n], act_fn_hidden='relu')
        self.dec_l = utils.fcnn(features=[dec_inps_n, 64, 64, dec_outs_n], act_fn_hidden='relu')

        self.funs_dyn_mat = None

    def fit(self, timeseries, global_only: bool=False, fixed_alpha: float=0.5):

        alpha = 1. if global_only else fixed_alpha

        # --! forward input timeseries to the main algorithm to get fit results
        outs = self.forward(timeseries, alpha)

        funs_g          = outs[0]
        funs_g_pred     = outs[1]
        funs_l          = outs[2]
        funs_l_pred     = outs[3]
        timeseries_pred = outs[4]

        loss_pred   = self._fit_prediction(timeseries, timeseries_pred)

        loss_lin_g  = self.fit_weight_lin_global * self._fit_linearity_g(funs_g, funs_g_pred)
        loss_lin_l  = self.fit_weight_lin_local * self._fit_linearity_l(funs_l, funs_l_pred)

        loss = loss_pred + loss_lin_g + loss_lin_l
        return loss, loss_pred, loss_lin_g, loss_lin_l

    def forward(self, timeseries, alpha: float=0.5):

        timesteps_dim = 1
        timesteps_n   = timeseries.shape[timesteps_dim]

        if timesteps_n % self.fun_params_kern_sz:
            print('err >> the size of input timeseries is not a multiple of a filter size')
            return

        # --! derive nonlinear function embeddings for given timeseries, such that
        # --! the output embeddings are shaped as [B, T / kern_sz = number of slices, funs_n]
        funs_g = self._embed_functions_g(timeseries)
        funs_l = self._embed_functions_l(timeseries)

        # --! we take the number of timeseries slices as a prediction horizon
        horizon = funs_g.shape[1]

        # --! perform global and local predictions
        funs_g_pred = self._predict_globally(funs_g, horizon)
        funs_l_pred = self._predict_locally(funs_l, horizon)

        # --! concatenate predicted parts with initial condictions to get full trajectories
        funs_g_pred = torch.cat([funs_g[:, :1], funs_g_pred], dim=1)
        funs_l_pred = torch.cat([funs_l[:, :1], funs_l_pred], dim=1)

        timeseries_pred_g  = self.dec_g(funs_g_pred)
        timeseries_pred_g  = timeseries_pred_g.reshape(timeseries_pred_g.shape[0], -1, self.timeseries_dims_n)

        timeseries_pred_l  = self.dec_l(funs_l_pred)
        timeseries_pred_l  = timeseries_pred_l.reshape(timeseries_pred_l.shape[0], -1, self.timeseries_dims_n)

        # --! we use addition operation to combine the global and local predictions
        timeseries_pred = alpha * timeseries_pred_g + (1 - alpha) * timeseries_pred_l

        # --! predicted embeddings are decoded back to timeseries
        #
        # --! and since timeseries are reconstructed as rows their shape is [B, C, T],
        # --! but the input timeseries are shaped as columns,
        # --! so reshape the reconstructed ones

        return funs_g, funs_g_pred, funs_l, funs_l_pred, timeseries_pred

    def _embed_functions_g(self, timeseries):

        # --! reshape input timeseries to work with dynamic kernels
        #
        # --! the length of timeseries is reshaped to extract the kernels
        # --! into the prelast dimension, i.e.
        # --! [B, T, C] -> [B, T / kern_sz, T_kern, C], where
        # --! B, T and C are the number of batch elements,
        # --! timesteps and data channels, respectively.
        #
        # --! note that -1 below infers the size of a dimension
        inps = timeseries.reshape(timeseries.shape[0], -1, self.fun_params_kern_sz, self.timeseries_dims_n)
        inps = inps.reshape(timeseries.shape[0], -1, self.fun_params_kern_sz * self.timeseries_dims_n)

        # --! based on inputs, encode parameter kernels (filters)
        kerns = self.fun_params_kern_enc_g(inps)

        all_params_n = sum(self.funs.values())

        # --! reshape encoded kernels to format their multiplication with inputs, e.g.
        # --!
        # --! kernels: [B, T / kern_sz, fun_params_n * kern_sz * C] ->
        # --! [B, T / kern_sz, fun_params_n, kern_sz, C]
        # --!
        # --! inputs: [B, T / kern_sz, kern_sz * C] ->
        # --! [B, T / kern_sz, kern_sz, C]
        kerns = kerns.reshape(
            inps.shape[0], inps.shape[1],
            all_params_n,
            self.fun_params_kern_sz * self.timeseries_dims_n)
        kerns = kerns.reshape(
            inps.shape[0], inps.shape[1],
            all_params_n,
            self.fun_params_kern_sz, self.timeseries_dims_n)
        inps = inps.reshape(inps.shape[0], inps.shape[1], self.fun_params_kern_sz, self.timeseries_dims_n)

        # --! with the help of kernels extract function parameters from input timeseries
        params = torch.einsum("blkdf, bldf -> blfk", kerns, inps)

        # --! split extracted parameters based on number of parameters required by each basis function
        params = torch.split(params, list(self.funs.values()), dim=-1)

        # --! measure nonlinear functions, or so-called embeddings, for the current slices of timeseries
        #
        # --! note that there is one single measurement of each function to describe
        # --! a slice, so the granularity of the slicing plays an important a role
        funs = torch.cat([self._meas_fun(fun, param) for fun, param in zip(self.funs.keys(), params)], dim=-1)

        # --! reshape dimensions to go from a shape [B, 1, C, funs_n]
        # --! to [B, 1, C * func_n]
        #
        # --! note that 1 denotes the currently iterated slice
        return funs.reshape(funs.shape[0], funs.shape[1], -1)

    def _embed_functions_l(self, timeseries):

        # --! reshape input timeseries to work with dynamic kernels
        #
        # --! the length of timeseries is reshaped to extract the kernels
        # --! into the prelast dimension, i.e.
        # --! [B, T, C] -> [B, T / kern_sz, T_kern, C], where
        # --! B, T and C are the number of batch elements,
        # --! timesteps and data channels, respectively.
        #
        # --! note that -1 below infers the size of a dimension
        inps = timeseries.reshape(timeseries.shape[0], -1, self.fun_params_kern_sz, self.timeseries_dims_n)
        inps = inps.reshape(timeseries.shape[0], -1, self.fun_params_kern_sz * self.timeseries_dims_n)

        # --! based on inputs, encode parameter kernels (filters)
        kerns = self.fun_params_kern_enc_l(inps)

        all_params_n = sum(self.funs.values())

        # --! reshape encoded kernels to format their multiplication with inputs, e.g.
        # --!
        # --! kernels: [B, T / kern_sz, fun_params_n * kern_sz * C] ->
        # --! [B, T / kern_sz, fun_params_n, kern_sz, C]
        # --!
        # --! inputs: [B, T / kern_sz, kern_sz * C] ->
        # --! [B, T / kern_sz, kern_sz, C]
        kerns = kerns.reshape(
            inps.shape[0], inps.shape[1],
            all_params_n,
            self.fun_params_kern_sz * self.timeseries_dims_n)
        kerns = kerns.reshape(
            inps.shape[0], inps.shape[1],
            all_params_n,
            self.fun_params_kern_sz, self.timeseries_dims_n)
        inps = inps.reshape(inps.shape[0], inps.shape[1], self.fun_params_kern_sz, self.timeseries_dims_n)

        # --! with the help of kernels extract function parameters from input timeseries
        params = torch.einsum("blkdf, bldf -> blfk", kerns, inps)

        # --! split extracted parameters based on number of parameters required by each basis function
        params = torch.split(params, list(self.funs.values()), dim=-1)

        # --! measure nonlinear functions, or so-called embeddings, for the current slices of timeseries
        #
        # --! note that there is one single measurement of each function to describe
        # --! a slice, so the granularity of the slicing plays an important a role
        funs = torch.cat([self._meas_fun(fun, param) for fun, param in zip(self.funs.keys(), params)], dim=-1)

        # --! reshape dimensions to go from a shape [B, 1, C, funs_n]
        # --! to [B, 1, C * func_n]
        #
        # --! note that 1 denotes the currently iterated slice
        return funs.reshape(funs.shape[0], funs.shape[1], -1)

    def _predict_globally(self, funs, horizon):

        # --! extract a matrix for global dynamics
        #
        # --! the matrix is unsqueezed to a shape [1, funs_n, funs_n] to allow
        # --! broadcasting when multiplying with functions
        timeseries_dyn_mat = torch.unsqueeze(self.timeseries_dyn.weight, 0)

        # --! raise local attention matrices to powers covering all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, funs_n, funs_n], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that dynamics matrices are also transposed to allow multiplication with functions
        timeseries_dyn_mat_powers = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(timeseries_dyn_mat, i), -2, -1) for i in range(1, horizon)], dim=1)

        # --! extract the initial conditions of function time (per slice) trajectories
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, funs_n] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        funs_ic = torch.unsqueeze(funs[:, :1], -2)

        # --! predict the global (per timeseries) evolution of functions by multiplying the initial conditions of
        # --! their trajectories by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, funs_n], i.e.
        # --! batches with functions trajectories consisting of inidividual function-points [1, funs_n]
        funs_pred_global = torch.matmul(funs_ic, timeseries_dyn_mat_powers)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        return torch.squeeze(funs_pred_global, -2)

    def _predict_locally(self, funs, horizon):

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
        self.funs_dyn_mat = funs_dyn_mat

        # --! raise local attention matrices to powers covering all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, funs_n, funs_n], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that dynamics matrices are also transposed to allow multiplication with functions
        # --! shaped as rows, i.e. [1, funs_n]
        funs_dyn_mat_powers = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(funs_dyn_mat, i), -2, -1) for i in range(1, horizon)], dim=1)

        # --! extract the initial conditions of function time (per slice) trajectories
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, funs_n] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        funs_ic = torch.unsqueeze(funs[:, :1], -2)

        # --! predict the local (per slice) evolution of functions by multiplying the initial conditions of
        # --! their trajectories by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, funs_n], i.e.
        # --! batches with functions trajectories consisting of inidividual function-points [1, funs_n]
        funs_pred_local = torch.matmul(funs_ic, funs_dyn_mat_powers)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        return torch.squeeze(funs_pred_local, -2)

    def _attention2entropy(self, attention):

        # --! to prevent log(0) we add a small offset to given attention matrix values
        epsilon   = 1e-10
        attention = attention + epsilon

        # --! we compute entropy inside attention rows, i.e. inside queries
        query_entropy = -torch.sum(attention * torch.log(attention), dim=-1)

        # --! computed query entropies are averaged
        return torch.mean(query_entropy)

    def _entropy2alpha(self, entropy, low=0.0, high=1.0, max_entropy=1.4):

        entropy = torch.clamp(entropy, 0.0, max_entropy)
        alpha = 1.0 - (entropy / max_entropy)  # inverse mapping
        return alpha * (high - low) + low

    def _meas_fun(self, fun, param):
        if fun == 'sin':
            return self._meas_sin(param)
        elif fun == 'cos':
            return self._meas_cos(param)
        elif fun == 'exp':
            return self._meas_exp(param)
        elif 'data' in fun:
            return self._meas_data(param)
        else:
            raise Exception("unsupported basis function!")

    def _meas_sin(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.sin(self.timeseries_timestep * freq)

    def _meas_cos(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.cos(self.timeseries_timestep * freq)

    def _meas_data(self, params):
        return params

    def _meas_sin2x(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.sin(2 * self.timeseries_timestep * freq)

    def _meas_cos2x(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.cos(2 * self.timeseries_timestep * freq)

    def _meas_exp(self, params):
        power = params
        return torch.exp(power)

    def _fit_embedding(self, timeseries, timeseries_recon):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(timeseries_recon, timeseries)

    def _fit_prediction(self, timeseries, timeseries_pred):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(timeseries_pred, timeseries)

    def _fit_linearity_g(self, funs, funs_pred):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(funs_pred, funs)

    def _fit_linearity_l(self, funs, funs_pred):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(funs_pred, funs)

