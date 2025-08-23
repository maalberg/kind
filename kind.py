# --!-------------------------------------------------------------------!
# --! Implementation of Kalman-inspired neural decomposition, or KIND
# --!-------------------------------------------------------------------!

import torch

from abc import abstractmethod
from abc import ABC as interface

from dataclasses import dataclass

import utils_data
import utils_nn


class operator(torch.nn.Module, interface):
    """Models dynamics of timeseries."""

    sin_nparam  = 2
    cos_nparam  = 2
    data_nparam = 1
    poly_nparam = 1
    exp_nparam  = 1

    def __init__(self, config):
        super().__init__()

        # --! store mutual configuration inside this base class
        self.timeseries_ndim       = config.timeseries_ndim
        self.timestep              = config.timestep
        self.lookback_nsample      = config.lookback_nsample
        self.forecast_nsample      = config.forecast_nsample

    @abstractmethod
    def embed(self, timeseries):
        """Embeds ``timeseries`` as functions of a latent space."""
        return

    @abstractmethod
    def predict_mean(self, functions):
        """Predicts ``functions`` embedded in a latent space."""
        return

    @abstractmethod
    def predict_var(self, errors):
        """Predicts ``errors`` of functions embedded in a latent space."""
        return

    @abstractmethod
    def forward(self, timeseries):
        """Predicts given ``timeseries`` based on an embedded function space."""
        return

    @abstractmethod
    def freeze_mean(self):
        """Makes submodules that are responsible for mean signal prediction fixed, untrainable."""
        return

    @abstractmethod
    def freeze_var(self):
        """Makes submodules that are responsible for the variance of prediction fixed, untrainable."""
        return

    @abstractmethod
    def unfreeze(self):
        """Makes all trainable submodules trainable again."""
        return

    def _respec_fun(self, spec):
        """Adapts function specifications ``spec`` to facilitate model internal workings.

        The adaptation converts the number of specific functions, e.g., 'sin', into
        the total number of parameters required by these specific functions. For
        example, if 5 functions of 'sin' are to be used, then the total number
        of parameters becomes 5 * 2 = 10, as a 'sin' takes 2 parameters.
        """

        newspec = spec.copy()

        for fun in newspec:
            if fun == 'sin':
                newspec[fun] = newspec[fun] * self.sin_nparam # sine takes two parameters
            elif fun == 'cos':
                newspec[fun] = newspec[fun] * self.cos_nparam # cosine takes two parameters

        return newspec

    def _eval_fun(self, fun, param):
        if fun == 'sin':
            return self._eval_sin(param)
        elif fun == 'cos':
            return self._eval_cos(param)
        elif fun == 'data':
            return self._eval_data(param)
        elif fun == 'poly':
            return self._eval_poly(param)
        elif fun == 'exp':
            return self._eval_exp(param)
        else:
            raise Exception("unsupported basis function!")

    def _eval_sinx(self, param, n):
        amp, ang = torch.split(param, 1, dim=-1)
        return amp * torch.sin(n * ang)

    def _eval_sin(self, param):
        param = torch.split(param, self.sin_nparam, dim=-1)
        mult  = torch.arange(len(param)) + 1
        return torch.cat([self._eval_sinx(p, m) for p, m in zip(param, mult)], dim=-1)

    def _eval_cosx(self, param, n):
        amp, ang = torch.split(param, 1, dim=-1)
        return amp * torch.cos(n * ang)

    def _eval_cos(self, param):
        param   = torch.split(param, self.cos_nparam, dim=-1)
        factor  = torch.arange(len(param)) + 1
        return torch.cat([self._eval_cosx(p, f) for p, f in zip(param, factor)], dim=-1)

    def _eval_data(self, param):
        return param

    def _eval_poly(self, param):
        param = torch.split(param, self.poly_nparam, dim=-1)
        degree = torch.arange(len(param)) + 1
        return torch.cat([p**d for p, d in zip(param, degree)], dim=-1)

    def _eval_exp(self, param):
        return torch.exp(param)


class operator_stationary(operator):
    """Models dynamics of stationary timeseries in a DMD-like manner."""

    def __init__(self, config):

        # --! initialize common operator parameters
        super().__init__(config)

        # --! this here is a convenient place to get the total number of functions, because
        # --! later the function configuration becomes respecified
        # --! to facilitate other things
        nfun = sum(config.fun_stat.values())

        # --! initialize stationary-specific parameters
        #
        # --! note that here the function configuration becomes respecified
        self.fun            = self._respec_fun(config.fun_stat)
        self.param_kernsize = config.param_kernsize_stat

        if self.forecast_nsample % self.param_kernsize:
            raise Exception('the number of forecast samples must be a multiple of a kernel size!')
        self.fun_nsample_forecast  = self.forecast_nsample // self.param_kernsize

        # --! here the function configuration is already respecified, such that it is convenient
        # --! to get the total number of required function parameters
        nparam      = sum(self.fun.values())
        fun_nsample = self.lookback_nsample // self.param_kernsize

        # --! create an encoder to encode timeseries into embedded functions
        #
        # --! more precisely, input timeseries are partitioned into slices, and the encoder
        # --! encodes slice-specific kernels for every function parameter
        # --! in the embedded (latent) space
        fun_enc_ni   = self.param_kernsize * self.timeseries_ndim
        fun_enc_no   = nparam * self.param_kernsize * self.timeseries_ndim
        self.fun_enc = utils_nn.fcnn(feat=[fun_enc_ni, 64, 64, fun_enc_no], actfun_hid='relu')

        # --! this linear transformation is supposed to prune the dimensionality of the
        # --! basis functions, such that only the number of these basis functions
        # --! influences the order of the DMD matrix, whereas the number of data
        # --! dimensions has no effect on the order
        fun_prune_ni = nfun * self.timeseries_ndim
        fun_prune_no = nfun
        self.fun_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)

        # --! create a DMD-like model (a matrix) that captures stationary (mean) dynamics
        #
        # --! since this matrix is learned only once and does not adapt during runtime,
        # --! this operator can also be called static, instead of stationary
        mod_mean_ni = nfun
        mod_mean_no = nfun
        self.mod_mean = torch.nn.Linear(mod_mean_ni, mod_mean_no, bias=False)

        # --! create a DMD-like model to capture stationary error dynamics
        mod_var_ni = nfun
        mod_var_no = nfun
        self.mod_var = torch.nn.Linear(mod_var_ni, mod_var_no, bias=False)

        # --! create prediction decoders to decode predicted embeddings back to timeseries and uncertainty
        pre_dec_ni        = nfun
        pre_dec_no        = self.param_kernsize * self.timeseries_ndim
        self.pre_mean_dec = utils_nn.fcnn(feat=[pre_dec_ni, 64, 64, pre_dec_no], actfun_hid='relu')
        self.pre_var_dec  = utils_nn.fcnn(feat=[pre_dec_ni, 64, 64, pre_dec_no], actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.timeseries_ndim)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.timeseries_ndim)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.fun_enc(i)

        nparam = sum(self.fun.values())

        # --! reshape encoded kernels to support their next multiplication with inputs, e.g.
        # --!
        # --! kernels:
        # --! [B, T / kernsize, nparam * kernsize * C] -> [B, T / kernsize, nparam, kernsize, C]
        # --!
        # --! inputs:
        # --! [B, T / kernsize, kernsize * C] -> [B, T / kernsize, kernsize, C]
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize * self.timeseries_ndim)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.timeseries_ndim)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.timeseries_ndim)

        # --! with the help of kernels extract function parameters from input timeseries
        param = torch.einsum("blkdf, bldf -> blfk", kern, i)

        # --! split extracted parameters based on the number of parameters required by each
        # --! type of grouped basis functions
        param = torch.split(param, list(self.fun.values()), dim=-1)

        # --! evaluate embedding functions at each slice of timeseries
        fun = torch.cat([self._eval_fun(f, p) for f, p in zip(self.fun.keys(), param)], dim=-1)

        # --! reshape dimensions to go from a shape [B, T / kernsize, ndim, nfun]
        # --! to [B, T / kernsize, ndim * nfun]
        fun = fun.reshape(fun.shape[0], fun.shape[1], -1)

        # --! prune extra dimensionality caused by multidimensional data
        return self.fun_prune(fun)

    def predict_mean(self, functions):

        lookback_nsample = functions.shape[1]
        forecast_nsample = self.fun_nsample_forecast

        horizon = lookback_nsample + forecast_nsample

        # --! extract the matrix of stationary dynamics
        #
        # --! the matrix is unsqueezed to a shape [1, nfun, nfun] to allow
        # --! broadcasting when multiplying with functions
        mat = torch.unsqueeze(self.mod_mean.weight, 0)

        # --! stack together matrices raised to powers to cover all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, nfun, nfun], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that we omit matrix transpose here, relying on training to figure it out
        mat_power = torch.stack([
            torch.linalg.matrix_power(mat, power) for power in range(1, horizon)], dim=1)

        # --! extract the initial conditions of function values, i.e. the first slice
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        fun_ic = torch.unsqueeze(functions[:, :1], -2)

        # --! predict the stationary evolution of function values by multiplying their initial conditions
        # --! by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun], i.e.
        # --! batches with functions trajectories consisting of individual function-points [1, nfun]
        fun_pre = torch.matmul(fun_ic, mat_power)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        fun_pre = torch.squeeze(fun_pre, -2)

        return torch.split(fun_pre, [lookback_nsample - 1, forecast_nsample], dim=1)

    def predict_var(self, errors):

        lookback_nsample = errors.shape[1]
        forecast_nsample = self.fun_nsample_forecast

        horizon = lookback_nsample + forecast_nsample

        # --! extract the matrix of stationary dynamics
        #
        # --! the matrix is unsqueezed to a shape [1, nfun, nfun] to allow
        # --! broadcasting when multiplying with functions
        mat = torch.unsqueeze(self.mod_var.weight, 0)

        # --! stack together matrices raised to powers to cover all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, nfun, nfun], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that we omit matrix transpose here, relying on training to figure it out
        mat_power = torch.stack([
            torch.linalg.matrix_power(mat, power) for power in range(1, horizon)], dim=1)

        # --! extract the initial condition of error history
        #
        # --! the initial condition (ic) is shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by the matrices
        err_ic = torch.unsqueeze(errors[:, :1], -2)

        # --! predict the evolution of errors by multiplying their initial conditions by the matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun],
        # --! i.e. a batch with error trajectories consisting of individual error-points [1, nfun]
        err_pre = torch.matmul(err_ic, mat_power)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        err_pre = torch.squeeze(err_pre, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        err_nsample = errors.shape[1]
        return torch.split(err_pre, [err_nsample - 1, self.fun_nsample_forecast], dim=1)

    def forward(self, timeseries):

        # --! from given timeseries, embed latent function values and then predict these embeddings
        # --! starting from the first value (initial condition) upto a specified horizon
        fun                        = self.embed(timeseries)
        fun_pre, fun_pre_forecast  = self.predict_mean(fun)
        fun_pre                    = torch.cat([fun[:, :1], fun_pre], dim=1)

        # --! some constants for convenience
        batsize          = fun.shape[0]
        nfun             = fun.shape[-1]

        # --! compute a function prediction error (difference) and normalize the error in order to
        # --! facilitate subsequent learning of the error dynamics
        dfun         = fun_pre - fun
        dfun_mean    = torch.mean(dfun, dim=1, keepdim=True)
        dfun         = dfun - dfun_mean
        scaler       = utils_data.minmax_scaler(feature_range=(-1, 1))
        dfun         = scaler.fit_transform(dfun)

        # --! predict the evolution of a function error starting from the first error value upto
        # --! a specified horizon
        dfun_pre, dfun_pre_forecast = self.predict_var(dfun)
        dfun_pre                    = torch.cat([dfun[:, :1], dfun_pre], dim=1)

        # --! denormalize predicted errors to restore original magnitudes, which are essential for
        # --! decoding the right uncertainty magnitudes
        dfun_pre_unsca = scaler.inverse_transform(torch.cat([dfun_pre, dfun_pre_forecast], dim=1))
        dfun_pre_unsca = dfun_pre_unsca + dfun_mean

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        timeseries_pre_mean  = self.pre_mean_dec(torch.cat([fun_pre, fun_pre_forecast], dim=1))
        timeseries_pre_mean  = timeseries_pre_mean.reshape(timeseries_pre_mean.shape[0], -1, self.timeseries_ndim)

        # --! decode predicted and denormalized function errors to model uncertainty (variance)
        timeseries_pre_var   = self.pre_var_dec(dfun_pre_unsca)
        timeseries_pre_var   = timeseries_pre_var.reshape(timeseries_pre_var.shape[0], -1, self.timeseries_ndim)

        return timeseries_pre_mean, timeseries_pre_var, fun, fun_pre, dfun, dfun_pre

    def freeze_mean(self):
        utils_nn.freeze_module(self.fun_enc)
        utils_nn.freeze_module(self.fun_prune)
        utils_nn.freeze_module(self.mod_mean)
        utils_nn.freeze_module(self.pre_mean_dec)

    def freeze_var(self):
        utils_nn.freeze_module(self.mod_var)
        utils_nn.freeze_module(self.pre_var_dec)

    def unfreeze(self):
        utils_nn.unfreeze_module(self.fun_enc)
        utils_nn.unfreeze_module(self.fun_prune)
        utils_nn.unfreeze_module(self.mod_mean)
        utils_nn.unfreeze_module(self.mod_var)
        utils_nn.unfreeze_module(self.pre_mean_dec)
        utils_nn.unfreeze_module(self.pre_var_dec)


class operator_transient(operator):
    """Models dynamics of transient timeseries using a Transformer-based attention mechanism."""

    def __init__(self, config):

        # --! initialize common operator parameters
        super().__init__(config)

        # --! this here is a convenient place to get the total number of functions, because
        # --! later the function configuration becomes respecified
        # --! to facilitate other things
        nfun = sum(config.fun_trans.values())

        # --! initialize transient-specific parameters
        #
        # --! note that here the function configuration becomes respecified
        self.fun            = self._respec_fun(config.fun_trans)
        self.param_kernsize = config.param_kernsize_trans
        self.mean_att_used  = config.mean_att_used
        self.var_att_used   = config.var_att_used

        if self.forecast_nsample % self.param_kernsize:
            raise Exception('the number of forecast samples must be a multiple of a kernel size!')
        self.fun_nsample_forecast  = self.forecast_nsample // self.param_kernsize

        # --! here the function configuration is already respecified, such that it is convenient
        # --! to get the total number of required function parameters
        nparam = sum(self.fun.values())
        fun_nsample = self.lookback_nsample // self.param_kernsize

        # --! create an MLP-based encoder to encode timeseries into embedded functions
        #
        # --! more precisely, input timeseries are partitioned into slices, and the encoder
        # --! encodes slice-specific kernels for every function parameter
        # --! in the embedded (latent) space
        fun_enc_ni   = self.param_kernsize * self.timeseries_ndim
        fun_enc_no   = nparam * self.param_kernsize * self.timeseries_ndim
        self.fun_enc = utils_nn.fcnn(feat=[fun_enc_ni, 64, 64, fun_enc_no], actfun_hid='relu')

        # --! this linear transformation is supposed to prune the dimensionality of the
        # --! basis functions, such that only the number of these basis functions
        # --! influences the order of linear matrices and the number of data
        # --! dimensions has no effect on that order
        fun_prune_ni = nfun * self.timeseries_ndim
        fun_prune_no = nfun
        self.fun_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)

        if self.mean_att_used:
            # --! encoder network which learns to attend over slices of embedded function values
            mod_mean_att_enc_ni = nfun

            # --! the attention encoder is implemented in terms of a Transformer encoder network
            self.mod_mean_att_enc = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=mod_mean_att_enc_ni,
                    nhead=1,
                    dim_feedforward=128,
                    batch_first=True),
                num_layers=3,
                enable_nested_tensor=False)

        # --! an additional generator to create linear time-varying matrices from
        # --! encoded attention; these matrices enable piecewise-linear
        # --! predictions of mean embedding sequences
        #
        # --! the generator takes a flattened sequence of function values and
        # --! returns a flattened sequence of square matrices,
        # --! covering all prediction horizon
        mod_mean_gen_ni = fun_nsample * nfun
        mod_mean_gen_no = (fun_nsample + self.fun_nsample_forecast) * nfun * nfun
        self.mod_mean_gen = utils_nn.fcnn(feat=[mod_mean_gen_ni, 64, 64, mod_mean_gen_no], actfun_hid='relu')

        if self.var_att_used:
            # --! encoder network which learns to attend over slices of embedded function errors
            mod_var_att_enc_ni = nfun

            # --! the attention encoder is implemented in terms of a Transformer encoder network
            self.mod_var_att_enc = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=mod_var_att_enc_ni,
                    nhead=1,
                    dim_feedforward=128,
                    batch_first=True),
                num_layers=3,
                enable_nested_tensor=False)

        # --! create a generator that produces models capturing the evolution of variance (uncertainty)
        #
        # --! the generator takes a flattened sequence of function values and returns
        # --! a flattened sequence of square matrices
        mod_var_gen_ni = fun_nsample * nfun
        mod_var_gen_no = (fun_nsample + self.fun_nsample_forecast) * nfun * nfun
        self.mod_var_gen = utils_nn.fcnn(feat=[mod_var_gen_ni, 64, 64, mod_var_gen_no], actfun_hid='relu')

        # --! create MLP-based decoders to decode embeddings back to timeseries with uncertainty
        pre_dec_ni        = nfun
        pre_dec_no        = self.param_kernsize * self.timeseries_ndim
        self.pre_mean_dec = utils_nn.fcnn(feat=[pre_dec_ni, 64, 64, pre_dec_no], actfun_hid='relu')
        self.pre_var_dec  = utils_nn.fcnn(feat=[pre_dec_ni, 64, 64, pre_dec_no], actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.timeseries_ndim)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.timeseries_ndim)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.fun_enc(i)

        nparam = sum(self.fun.values())

        # --! reshape encoded kernels to support their next multiplication with inputs, e.g.
        # --!
        # --! kernels:
        # --! [B, T / kernsize, nparam * kernsize * C] -> [B, T / kernsize, nparam, kernsize, C]
        # --!
        # --! inputs:
        # --! [B, T / kernsize, kernsize * C] -> [B, T / kernsize, kernsize, C]
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize * self.timeseries_ndim)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.timeseries_ndim)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.timeseries_ndim)

        # --! with the help of kernels extract function parameters from input timeseries
        param = torch.einsum("blkdf, bldf -> blfk", kern, i)

        # --! split extracted parameters based on number of parameters required by each basis function
        param = torch.split(param, list(self.fun.values()), dim=-1)

        # --! evaluate embedding functions at each slice of timeseries
        #
        # --! note that there is one single measurement of each function to describe
        # --! a slice, so the granularity of slicing plays an important a role
        fun = torch.cat([self._eval_fun(f, p) for f, p in zip(self.fun.keys(), param)], dim=-1)

        # --! reshape dimensions to go from a shape [B, T / kernsize, ndim, nfun]
        # --! to [B, T / kernsize, ndim * nfun]
        fun = fun.reshape(fun.shape[0], fun.shape[1], -1)

        # --! prune extra dimensionality caused by multidimensional data
        return self.fun_prune(fun)

    def predict_mean(self, functions):

        batsize     = functions.shape[0]
        fun_nsample = functions.shape[1]
        nfun        = functions.shape[2]

        if self.mean_att_used:
            # --! encode attention over the sequence of function values
            #
            # --! functions, which are shaped as [B, T / kernzise, nfun] are encoded into attention with the same shape
            #
            # --! attention is produced for each sequence step (rows of attention matrix); this information can be
            # --! used to derive linear time-varying matrices that locally adapt to changes in dynamics
            functions = self.mod_mean_att_enc(functions)

        # --! note that we omit matrix transpose here, relying on training to figure it out
        #
        # --! note also that the neural network, which generates matrices, is already configured to
        # --! cover the entire prediction horizon, i.e. lookback and forecast windows
        mod_mean_gen_i = torch.flatten(functions, start_dim=1)
        mat            = self.mod_mean_gen(mod_mean_gen_i).reshape(batsize, -1, nfun, nfun)

        # --! accumulate matrix products to enable predictions, such as z2 = A1*z1, z3 = A2*A1*z1, etc,
        # --! where zi are our embeddings, and where Ai are linear time-varying matrices
        mat = utils_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial conditions of function values, i.e. the first slice
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        fun_ic = torch.unsqueeze(functions[:, :1], -2)

        # --! predict the stationary evolution of function values by multiplying their initial conditions
        # --! by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun], i.e.
        # --! batches with functions trajectories consisting of inidividual function-points [1, nfun]
        fun_pre = torch.matmul(fun_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        fun_pre = torch.squeeze(fun_pre, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(fun_pre, [fun_nsample - 1, self.fun_nsample_forecast], dim=1)

    def predict_var(self, errors):

        # --! constants for convenience
        batsize     = errors.shape[0]
        err_nsample = errors.shape[1]
        nfun        = errors.shape[2]

        if self.var_att_used:
            errors = self.mod_var_att_enc(errors)

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of error values
        i           = torch.flatten(errors, start_dim=1)
        mat         = self.mod_var_gen(i).reshape(batsize, -1, nfun, nfun)
        mat         = utils_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial condition of error history
        #
        # --! the initial condition (ic) is shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by the matrices
        err_ic      = torch.unsqueeze(errors[:, :1], -2)

        # --! predict the evolution of errors by multiplying their initial conditions by the matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun],
        # --! i.e. a batch with error trajectories consisting of individual error-points [1, nfun]
        err_pre = torch.matmul(err_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        err_pre = torch.squeeze(err_pre, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(err_pre, [err_nsample - 1, self.fun_nsample_forecast], dim=1)

    def forward(self, timeseries):

        # --! from given timeseries, embed latent function values and then predict these embeddings
        # --! starting from the first value (initial condition) upto a specified horizon
        fun                       = self.embed(timeseries)
        fun_pre, fun_pre_forecast = self.predict_mean(fun)
        fun_pre                   = torch.cat([fun[:, :1], fun_pre], dim=1)

        # --! compute a function prediction error (difference) and normalize the error in order to
        # --! facilitate subsequent learning of the error dynamics
        dfun         = fun_pre - fun
        dfun_mean    = torch.mean(dfun, dim=1, keepdim=True)
        dfun         = dfun - dfun_mean
        scaler       = utils_data.minmax_scaler(feature_range=(-1, 1))
        dfun         = scaler.fit_transform(dfun)

        # --! predict the evolution of a function error starting from the first error value upto
        # --! a specified horizon
        dfun_pre, dfun_pre_forecast = self.predict_var(dfun)
        dfun_pre                    = torch.cat([dfun[:, :1], dfun_pre], dim=1)

        # --! denormalize predicted errors to restore original magnitudes, which are essential for
        # --! decoding the right uncertainty magnitudes
        dfun_pre_unsca = scaler.inverse_transform(torch.cat([dfun_pre, dfun_pre_forecast], dim=1))
        dfun_pre_unsca = dfun_pre_unsca + dfun_mean

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        timeseries_pre_mean = self.pre_mean_dec(torch.cat([fun_pre, fun_pre_forecast], dim=1))
        timeseries_pre_mean = timeseries_pre_mean.reshape(timeseries_pre_mean.shape[0], -1, self.timeseries_ndim)

        # --! decode predicted and denormalized function errors to model uncertainty (variance)
        timeseries_pre_var  = self.pre_var_dec(dfun_pre_unsca)
        timeseries_pre_var  = timeseries_pre_var.reshape(timeseries_pre_var.shape[0], -1, self.timeseries_ndim)

        return timeseries_pre_mean, timeseries_pre_var, fun, fun_pre, dfun, dfun_pre

    def freeze_mean(self):
        utils_nn.freeze_module(self.fun_enc)
        utils_nn.freeze_module(self.fun_prune)
        if self.mean_att_used:
            utils_nn.freeze_module(self.mod_mean_att_enc)
        utils_nn.freeze_module(self.mod_mean_gen)
        utils_nn.freeze_module(self.pre_mean_dec)

    def freeze_var(self):
        if self.var_att_used:
            utils_nn.freeze_module(self.mod_var_att_enc)
        utils_nn.freeze_module(self.mod_var_gen)
        utils_nn.freeze_module(self.pre_var_dec)

    def unfreeze(self):
        utils_nn.unfreeze_module(self.fun_enc)
        utils_nn.unfreeze_module(self.fun_prune)
        if self.mean_att_used:
            utils_nn.unfreeze_module(self.mod_mean_att_enc)
        if self.var_att_used:
            utils_nn.unfreeze_module(self.mod_var_att_enc)
        utils_nn.unfreeze_module(self.mod_mean_gen)
        utils_nn.unfreeze_module(self.mod_var_gen)
        utils_nn.unfreeze_module(self.pre_mean_dec)
        utils_nn.unfreeze_module(self.pre_var_dec)


class run_mode(interface):
    """Represents the current running mode of a model, i.e. fit or evaluate."""

    def __init__(self, model):
        super().__init__()

        # --! reference to a model that we run
        self._model = model

    @abstractmethod
    def fit_next(self):
        """
        Advances a model to the next fit phase, provided the next phase is avalable.
        Returns True if the model has been successfully advanced.
        """
        return

    @abstractmethod
    def fit(self, param):
        """Fits a model parameterized by ``param``."""
        return

    @abstractmethod
    def forward(self, timeseries):
        """Forwards given ``timeseries`` to the current mode algorithm."""
        return

    def _forward(self, timeseries):

        # --! execute both operators on given time series
        timeseries_stat, timeseries_stat_logvar, fun_stat, fun_stat_pre, _, _ = self._model.operator_stat(timeseries)
        timeseries_trans, timeseries_trans_logvar, fun_trans, fun_trans_pre, _, _ = self._model.operator_trans(timeseries)

        # --! derive alpha
        timeseries_stat_var  = torch.exp(timeseries_stat_logvar) + 1e-6
        timeseries_trans_var = torch.exp(timeseries_trans_logvar) + 1e-6
        alpha = timeseries_trans_var / (timeseries_trans_var + timeseries_stat_var)

        # --! blend the two types of time series using the derived alpha to get the final prediction
        timeseries_pre = alpha * timeseries_stat + (1 - alpha) * timeseries_trans

        o = (
            timeseries_pre,
            timeseries_stat, timeseries_stat_logvar,
            timeseries_trans, timeseries_trans_logvar,
            fun_stat, fun_stat_pre,
            fun_trans, fun_trans_pre,
            alpha
        )

        return o


class mode_eval(run_mode):
    """Represents the evaluation mode of a model."""

    def __init__(self, model):
        super().__init__(model)

    def fit_next(self):
        print('wrn >> model is in evaluation mode')
        return False

    def fit(self, param):
        print('wrn >> model is in evaluation mode')
        return None

    def forward(self, timeseries):
        """Forecasts given ``timeseries`` in evaluation mode.

        The ``timeseries`` are expected to represent an unnormalized lookback window. The window is
        then normalized and forecasted by a number of steps configured in
        model parameter ``forecast_nsample``.
        """

        # --! remove mean
        mean = torch.mean(timeseries, dim=1, keepdim=True)
        timeseries = timeseries - mean

        # --! scale data using minmax to a range from -1 to 1
        scaler = utils_data.minmax_scaler(feature_range=(-1, 1))
        timeseries = scaler.fit_transform(timeseries)

        o = self._forward(timeseries)

        # --! extract to-be-unscaled timeseries from the forwarded result
        timeseries_pre       = o[0]
        timeseries_stat      = o[1]
        timeseries_trans     = o[3]

        # --! unscale resulting timeseries
        timeseries_pre       = scaler.inverse_transform(timeseries_pre)
        timeseries_pre       = timeseries_pre + mean
        timeseries_stat      = scaler.inverse_transform(timeseries_stat)
        timeseries_stat      = timeseries_stat + mean
        timeseries_trans     = scaler.inverse_transform(timeseries_trans)
        timeseries_trans     = timeseries_trans + mean

        # --! put unscaled timeseries back to the result tuple and return the tuple
        o    = list(o)
        o[0] = timeseries_pre
        o[1] = timeseries_stat
        o[3] = timeseries_trans

        return tuple(o)


class mode_fit(run_mode):
    """Represents the fit mode of a model."""

    def __init__(self, model):
        super().__init__(model)

    def fit_next(self):
        return self._model._fit_phase.next()

    def fit(self, param):

        # --! first of all, enter the current fit phase to initialize required parameters
        self._model._fit_phase.enter(param)

        # --! prepare test data
        testdata    = utils_data.read_datafile(f'{self._model._fit_phase.datadir}/valid', self._model._fit_phase.timeseries_nsample)
        testdataset = torch.utils.data.TensorDataset(testdata)

        # --! specify an optimizer for fit
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=self._model._fit_phase.learnrate,
            weight_decay=self._model._fit_phase.weightdecay)

        trainloss_predict    = []
        trainloss_sta_lin    = []
        trainloss_dyn_lin    = []

        # --! training duration
        if self._model._fit_phase.isverbose: print(f"inf >> number of data files for training is {self._model._fit_phase.train_nfile}")

        for ifile in range(self._model._fit_phase.train_nfile):
            if self._model._fit_phase.isverbose: print(f"inf >> processing training file number {ifile + 1}")

            # --! prepare training data
            traindata    = utils_data.read_datafile(
                f'{self._model._fit_phase.datadir}/train{ifile + 1}',
                self._model._fit_phase.timeseries_nsample)
            traindataset = torch.utils.data.TensorDataset(traindata)
            traindatafun = torch.utils.data.DataLoader(traindataset, batch_size=self._model._fit_phase.batsize, shuffle=True)

            # --! train
            for epoch in range(self._model._fit_phase.nepoch):
                for data in traindatafun:
                    # --! take all channels of these time series for training
                    timeseries = data[0]
                    optimizer.zero_grad()

                    # --! fit a model to training time series
                    loss, loss_predict, loss_lin_g, loss_lin_l = self._model._fit_phase.compute_loss(timeseries)

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        trainloss_predict.append(loss_predict)
                        trainloss_sta_lin.append(loss_lin_g)
                        trainloss_dyn_lin.append(loss_lin_l)

        o = (
            trainloss_predict,
            trainloss_sta_lin, trainloss_dyn_lin
        )
        return o

    def forward(self, timeseries):
        """Predicts given ``timeseries`` in fit mode.

        The ``timeseries`` are expected to be scaled to range from -1 to 1 with mean removed.
        """
        return self._forward(timeseries)


class fit_phase(interface):
    """Manages a certain phase of a model fit, e.g. fit a stationary mean, or fit a transient variance."""

    def __init__(self, model):
        super().__init__()

        # --! reference to model that we fit
        self._model = model

        # --! create placeholders for phase parameters
        self.timeseries_nsample    = None
        self.train_nfile           = None
        self.nepoch                = None
        self.batsize               = None
        self.alphafun              = None
        self.learnrate             = None
        self.weightdecay           = None
        self.isverbose             = None

        self.datadir               = None

    @abstractmethod
    def enter(self, param):
        """Enters this phase and parameterizes it using ``param``."""

        # --! initialize phase parameters
        self.timeseries_nsample    = param['timeseries_nsample']
        self.train_nfile           = param['train_nfile']
        self.nepoch                = param['nepoch']
        self.batsize               = param['batsize']
        self.learnrate             = param['learnrate']
        self.weightdecay           = param['weightdecay']
        self.isverbose             = param['isverbose']

    @abstractmethod
    def compute_loss(self, timeseries):
        """Use given ``timeseries`` to compute the loss of this fit phase."""
        return

    @abstractmethod
    def next(self) -> bool:
        """
        Transitions to the next phase, provided it is available.
        Returns True if the transition takes place,
        returns False otherwise.
        """
        return

    def _compute_varloss(self, timeseries, timeseries_predict_mean, timeseries_predict_var):
        logvar = torch.clamp(timeseries_predict_var, min=-5, max=5)
        var    = torch.exp(logvar) + 1e-6

        loss_fun = torch.nn.GaussianNLLLoss()
        return loss_fun(timeseries_predict_mean, timeseries, var)

    def _compute_meanloss(self, timeseries, timeseries_pred):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(timeseries_pred, timeseries)


class phase_stationary_mean(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering stationary mean phase")

        super().enter(param)

        self._model.operator_trans.freeze_mean()
        self._model.operator_trans.freeze_var()

        self._model.operator_stat.unfreeze()
        self._model.operator_stat.freeze_var()

        self.datadir = param['stadatadir']

    def compute_loss(self, timeseries):

        lookback = timeseries[:, :self._model.lookback_nsample]
        timeseries_pre_mean, _, fun, fun_pre, _, _ = self._model.operator_stat(lookback)

        loss_recon  = self._compute_meanloss(timeseries, timeseries_pre_mean)
        loss_linear = self._compute_meanloss(fun, fun_pre)

        loss = loss_recon + loss_linear

        return loss, loss_recon, loss_linear, 0.

    def next(self):
        self._model._set_phase(self._model._get_phase_stat_var())
        return True


class phase_stationary_var(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering stationary variance phase")

        super().enter(param)

        self._model.operator_trans.freeze_mean()
        self._model.operator_trans.freeze_var()

        self._model.operator_stat.unfreeze()
        self._model.operator_stat.freeze_mean()

        self.datadir = param['mixdatadir']

    def compute_loss(self, timeseries):

        lookback = timeseries[:, :self._model.lookback_nsample]
        timeseries_pre_mean, timeseries_pre_var, _, _, dfun, dfun_pre = self._model.operator_stat(lookback)

        loss_linear = self._compute_meanloss(dfun, dfun_pre)
        loss_recon  = self._compute_varloss(timeseries, timeseries_pre_mean, timeseries_pre_var)

        loss = loss_recon + loss_linear

        return loss, loss_recon, loss_linear, 0.

    def next(self):
        self._model._set_phase(self._model._get_phase_trans_mean())
        return True


class phase_transient_mean(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering transient mean phase")

        super().enter(param)

        self._model.operator_trans.unfreeze()
        self._model.operator_trans.freeze_var()

        self._model.operator_stat.freeze_mean()
        self._model.operator_stat.freeze_var()

        self.datadir = param['transdatadir']

    def compute_loss(self, timeseries):

        lookback = timeseries[:, :self._model.lookback_nsample]
        timeseries_pre_mean, _, fun, fun_pre, _, _ = self._model.operator_trans(lookback)

        loss_recon  = self._compute_meanloss(timeseries, timeseries_pre_mean)
        loss_linear = self._compute_meanloss(fun, fun_pre)

        loss = loss_recon + loss_linear

        return loss, loss_recon, 0., loss_linear

    def next(self):
        self._model._set_phase(self._model._get_phase_trans_var())
        return True


class phase_transient_var(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering transient variance phase")

        super().enter(param)

        self._model.operator_trans.unfreeze()
        self._model.operator_trans.freeze_mean()

        self._model.operator_stat.freeze_mean()
        self._model.operator_stat.freeze_var()

        self.datadir = param['mixdatadir']

    def compute_loss(self, timeseries):

        lookback = timeseries[:, :self._model.lookback_nsample]
        timeseries_pre_mean, timeseries_pre_var, _, _, dfun, dfun_pre = self._model.operator_trans(lookback)

        loss_linear = self._compute_meanloss(dfun, dfun_pre)
        loss_recon  = self._compute_varloss(timeseries, timeseries_pre_mean, timeseries_pre_var)
        loss        = loss_linear + loss_recon

        return loss, loss_recon, loss_linear, 0.

    def next(self):
        return False


@dataclass
class model_config:
    """
    Stores KIND model configuration.
    """

    # --! number of dimensions in time series
    timeseries_ndim: int
 
    # --! timestep that was used to sample timeseries
    timestep: float

    # --! number of time series samples in a lookback window
    lookback_nsample: int

    # --! number of time series samples in a forecast window
    forecast_nsample: int

    # --! basis functions used to build lifted embeddings for stationary and transient operators
    #
    # --! these dictionaries are structured as
    #
    # --!  *  key:   function name [str]
    # --!  *  value: number of functions of this type [int]
    fun_stat: dict
    fun_trans: dict

    # --! size of dynamic parameter filters encoded by an encoder from timeseries data
    #
    # --! Timeseries are partioned into slices that are encoded by an encoder to
    # --! produce, so to say, dynamic kernels, or filters. These kernels
    # --! help extract specific features, i.e. nonlinear function
    # --! parameters from the raw timeseries. In constrast to
    # --! static kernels in convolutional neural networks,
    # --! the kernels here are dynamic, because they
    # --! are produced from the timeseries every time.
    #
    # --! there are two sizes: one dedicated to a stationary operator,
    # --! the other - to a transient one.
    param_kernsize_stat: int
    param_kernsize_trans : int

    # --! flags that enable attention in transient prediction routines
    mean_att_used: bool
    var_att_used: bool


class model(torch.nn.Module):
    """Models Kalman-inpired neural decomposition, or KIND.

    This model captures the evolution of timeseries by first decomposing them into stationary and
    transient components, predicting these components into the future and
    finally blending the predictions in a Kalman-inspired manner.
    """

    def __init__(self, config):
        super().__init__()

        self.timeseries_ndim       = config.timeseries_ndim
        self.timestep              = config.timestep
        self.lookback_nsample      = config.lookback_nsample
        self.forecast_nsample      = config.forecast_nsample

        self.operator_stat = operator_stationary(config)
        self.operator_trans = operator_transient(config)

        self._fit_phase_stat_mean  = phase_stationary_mean(self)
        self._fit_phase_stat_var   = phase_stationary_var(self)
        self._fit_phase_trans_mean = phase_transient_mean(self)
        self._fit_phase_trans_var  = phase_transient_var(self)
        self._fit_phase            = self._fit_phase_stat_mean

        self._mode_fit  = mode_fit(self)
        self._mode_eval = mode_eval(self)
        self._mode      = self._mode_fit

    def fit_next(self):
        return self._get_mode().fit_next()

    def fit(self, param):
        return self._get_mode().fit(param)

    def forward(self, timeseries):
        return self._get_mode().forward(timeseries)

    def train(self, mode: bool = True):
        """If ``mode`` is True, switches this model into fit mode.
        Otherwise, if ``mode`` is False, switches this model into evaluation mode.
        """

        if mode is True:
            self._set_mode(self._get_mode_fit())
        else:
            self._set_mode(self._get_mode_eval())

        # --! call the superclass to finish the mode switch
        return super().train(mode)

    def eval(self):
        """Sets the model in evaluation mode."""

        # --! the superclass is called from the train method, so just delegate execution
        return self.train(False)

    def _get_mode_fit(self):
        return self._mode_fit

    def _get_mode_eval(self):
        return self._mode_eval

    def _get_mode(self):
        return self._mode

    def _set_mode(self, mode):
        self._mode = mode

    def _get_phase_stat_mean(self):
        return self._fit_phase_stat_mean

    def _get_phase_stat_var(self):
        return self._fit_phase_stat_var

    def _get_phase_trans_mean(self):
        return self._fit_phase_trans_mean

    def _get_phase_trans_var(self):
        return self._fit_phase_trans_var

    def _get_phase(self):
        return self._fit_phase

    def _set_phase(self, phase):
        self._fit_phase = phase

