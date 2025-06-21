import torch

from abc import abstractmethod
from abc import ABC as interface

from dataclasses import dataclass

import utilities as utils


class operator(torch.nn.Module, interface):
    """Models dynamics of timeseries."""

    def __init__(self, config):
        super().__init__()

        # --! store mutual configuration inside this base class
        self.timeseries_ndim       = config.timeseries_ndim
        self.timeseries_nsample    = config.timeseries_nsample
        self.timestep              = config.timestep
        self.fun                   = config.fun
        self.param_kernsize        = config.param_kernsize

    @abstractmethod
    def embed(self, timeseries):
        """Embeds ``timeseries`` as functions of a latent space."""
        return

    @abstractmethod
    def predict(self, functions):
        """Predicts ``functions`` embedded in a latent space."""
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
        """Makes trainable submodules trainable again."""
        return

    def _eval_fun(self, fun, param):
        if fun == 'sin':
            return self._eval_sin(param)
        elif fun == 'cos':
            return self._eval_cos(param)
        elif fun == 'exp':
            return self._eval_exp(param)
        elif 'data' in fun:
            return self._eval_data(param)
        elif 'poly' in fun:
            deg = utils.extract_poly_deg(fun)
            return self._eval_poly(param, deg)
        else:
            raise Exception("unsupported basis function!")

    def _eval_sin(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.sin(self.timestep * freq)

    def _eval_cos(self, params):
        amp, freq = torch.split(params, 1, dim=-1)
        return amp * torch.cos(self.timestep * freq)

    def _eval_data(self, params):
        return params

    def _eval_exp(self, params):
        power = params
        return torch.exp(power)

    def _eval_poly(self, params, deg):
        return torch.sum(torch.cat([params**(i+1) for i in range(deg)], dim=-1), -1, keepdim=True)


class operator_sta(operator):
    """Models dynamics of stationary timeseries in a dynamic mode decomposition-like manner."""

    def __init__(self, config):
        super().__init__(config)

        # --! derive some details of basis functions for convenience
        nfun   = len(self.fun)
        nparam = sum(self.fun.values())

        # --! create an MLP-based encoder to encode timeseries into embeddings
        #
        # --! more precisely, input timeseries are partitioned into slices, and the encoder
        # --! encodes slice-specific kernels for every function parameter
        # --! in the embedded (latent) space
        enc_ni   = self.param_kernsize * self.timeseries_ndim
        enc_no   = nparam * self.param_kernsize * self.timeseries_ndim
        self.enc = utils.fcnn(feat=[enc_ni, 64, 64, enc_no], actfun_hid='relu')

        # --! create a learnable DMD-like model (a matrix) that captures stationary dynamics
        #
        # --! since this matrix is learned only once and does not adapt during runtime,
        # --! this operator can also be called static, instead of stationary
        model_ni = nfun * self.timeseries_ndim
        model_no = nfun * self.timeseries_ndim
        self.model = torch.nn.Linear(model_ni, model_no, bias=False)

        # --! create MLP-based decoders to decode embeddings back to timeseries with uncertainty (mean and variance)
        #
        # --! features of the variance decoder are doubled to accommodate prediction errors
        dec_ni          = nfun * self.timeseries_ndim
        dec_no          = self.param_kernsize * self.timeseries_ndim
        self.dec_mean   = utils.fcnn(feat=[dec_ni, 64, 64, dec_no], actfun_hid='relu')
        self.dec_var    = utils.fcnn(feat=[dec_ni * 2, 64, 64, dec_no], actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, C] -> [B, T / kernsize, kernsize, C], where
        # --! B, T and C are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.timeseries_ndim)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.timeseries_ndim)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.enc(i)

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

        # --! reshape dimensions to go from a shape [B, 1, C, nfun]
        # --! to [B, 1, C * nfun]
        #
        # --! note that 1 denotes the currently iterated slice
        return fun.reshape(fun.shape[0], fun.shape[1], -1)

    def predict(self, functions):

        horizon = functions.shape[1]

        # --! extract the matrix of stationary dynamics
        #
        # --! the matrix is unsqueezed to a shape [1, nfun, nfun] to allow
        # --! broadcasting when multiplying with functions
        mat = torch.unsqueeze(self.model.weight, 0)

        # --! stack together matrices raised to powers to cover all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, nfun, nfun], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that dynamics matrices are also transposed to allow multiplication with functions
        mat_power = torch.stack([
            torch.transpose(
                torch.linalg.matrix_power(mat, power), -2, -1) for power in range(1, horizon)], dim=1)

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
        fun_predict = torch.matmul(fun_ic, mat_power)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        return torch.squeeze(fun_predict, -2)

    def forward(self, timeseries):
        fun         = self.embed(timeseries)
        fun_predict = self.predict(fun)

        # --! concatenate an initial condiction of a function with its predicted part to get a full trajectory
        fun_predict = torch.cat([fun[:, :1], fun_predict], dim=1)

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, C]
        timeseries_predict_mean  = self.dec_mean(fun_predict)
        timeseries_predict_mean  = timeseries_predict_mean.reshape(timeseries_predict_mean.shape[0], -1, self.timeseries_ndim)

        # --! to facilitate variance learning we explicitly provide prediction error along the prediction itself
        dfun                     = fun_predict - fun
        dec_var_i                = torch.cat([fun_predict, dfun], dim=-1)
        timeseries_predict_var   = self.dec_var(dec_var_i)
        timeseries_predict_var   = timeseries_predict_var.reshape(timeseries_predict_var.shape[0], -1, self.timeseries_ndim)

        return timeseries_predict_mean, timeseries_predict_var, fun, fun_predict

    def freeze_mean(self):
        utils.freeze_module(self.enc)
        utils.freeze_module(self.model)
        utils.freeze_module(self.dec_mean)

    def freeze_var(self):
        utils.freeze_module(self.dec_var)

    def unfreeze(self):
        utils.unfreeze_module(self.enc)
        utils.unfreeze_module(self.model)
        utils.unfreeze_module(self.dec_mean)
        utils.unfreeze_module(self.dec_var)


class operator_dyn(operator):
    """Models dynamics of transient timeseries using the attention mechanism."""

    def __init__(self, config):
        super().__init__(config)

        # --! derive some details of basis functions for convenience
        nfun   = len(self.fun)
        nparam = sum(self.fun.values())

        # --! create an MLP-based encoder to encode timeseries into embeddings
        #
        # --! more precisely, input timeseries are partitioned into slices, and the encoder
        # --! encodes slice-specific kernels for every function parameter
        # --! in the embedded (latent) space
        embed_enc_ni   = self.param_kernsize * self.timeseries_ndim
        embed_enc_no   = nparam * self.param_kernsize * self.timeseries_ndim
        self.embed_enc = utils.fcnn(feat=[embed_enc_ni, 64, 64, embed_enc_no], actfun_hid='relu')

        # --! encoder network, which produces adaptation context, acts along the sequence of function
        # --! values, but its features are the number of these embedding functions
        adapt_enc_ni = nfun

        # --! the encoder of adaptation context is implemented in terms of a Transformer network
        self.adapt_enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=adapt_enc_ni, nhead=1, dim_feedforward=128, batch_first=True),
            num_layers=3,
            enable_nested_tensor=False)

        # --! an additional MLP-based neural network to generate linear time-varying matrices from
        # --! encoded adaptation context; these matrices enable piecewise-linear
        # --! predictions of embedding sequences
        self.model_from = utils.fcnn(feat=[nfun, 64, 64, nfun*nfun], actfun_hid='relu')

        # --! create an MLP-based decoder to decode embeddings back to timeseries
        dec_ni   = nfun * self.timeseries_ndim
        dec_no   = self.param_kernsize * self.timeseries_ndim
        self.dec = utils.fcnn(feat=[dec_ni, 64, 64, dec_no], actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, C] -> [B, T / kernsize, kernsize, C], where
        # --! B, T and C are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.timeseries_ndim)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.timeseries_ndim)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.embed_enc(i)

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

        # --! reshape dimensions to go from a shape [B, 1, C, nfun]
        # --! to [B, 1, C * nfun]
        #
        # --! note that 1 denotes the currently iterated slice
        return fun.reshape(fun.shape[0], fun.shape[1], -1)

    def predict(self, functions):

        batsize = functions.shape[0]
        horizon = functions.shape[1]
        nfun    = functions.shape[-1]

        # --! encode adaptation context in the sequence of function values
        #
        # --! functions, which are shaped as [B, S, F] are encoded to a shape of [B, S, C],
        # --! where B and S are the number of batch elements batsize and sequence steps,
        # --! and where F and C are the number of functions nfun and
        # --! context channels, respectively
        adapt = torch.split(self.adapt_enc(functions), 1, dim=1)

        # --! using context for each sequence step, produce a sequence of linear time-varying matrices
        # --! that locally adapt to changes in dynamics
        mat = torch.cat([
            self.model_from(context).reshape(batsize, 1, nfun, nfun).transpose(-2, -1) for context in adapt], dim=1)

        # --! accumulate matrix products to enable predictions, such as z2 = A1*z1, z3 = A2*A1*z1, etc,
        # --! where zi are our embeddings, and where Ai are linear time-varying matrices
        #
        # --! we omit the last matrix in the sequence An, because there is no way to check the correctness
        # --! of the corresponding forecasted result z{n+1}
        mat = torch.cumprod(mat[:, :-1], dim=1)

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
        fun_predict = torch.matmul(fun_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        return torch.squeeze(fun_predict, -2)

    def forward(self, timeseries):
        fun         = self.embed(timeseries)
        fun_predict = self.predict(fun)

        # --! concatenate an initial condiction of a function with its predicted part to get a full trajectory
        fun_predict = torch.cat([fun[:, :1], fun_predict], dim=1)

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded timeseries back to their
        # --! original shape [B, T, C]
        timeseries_predict = self.dec(fun_predict)
        timeseries_predict = timeseries_predict.reshape(timeseries_predict.shape[0], -1, self.timeseries_ndim)

        return timeseries_predict, fun, fun_predict

    def freeze_mean(self):
        utils.freeze_module(self.embed_enc)
        utils.freeze_module(self.adapt_enc)
        utils.freeze_module(self.model_from)
        utils.freeze_module(self.dec)

    def freeze_var(self):
        raise NotImplementedError

    def unfreeze(self):
        utils.unfreeze_module(self.embed_enc)
        utils.unfreeze_module(self.adapt_enc)
        utils.unfreeze_module(self.model_from)
        utils.unfreeze_module(self.dec)


@dataclass
class detuning_config:
    """
    Stores detuning configuration.
    """

    # --! number of dimensions and samples in timeseries
    timeseries_ndim: int
    timeseries_nsample: int

    # --! timestep that was used to sample timeseries
    timestep: float

    # --! basis functions used to build lifted embedding
    #
    # --! this dictionary is structured as
    #
    # --!  *  function name: str
    # --!  *  number of function parameters: int
    fun: dict

    # --! size of dynamic parameter filters encoded by an encoder from timeseries data
    #
    # --! Timeseries are partioned into slices that are encoded by an encoder to
    # --! produce, so to say, dynamic kernels, or filters. These kernels
    # --! help extract specific features, i.e. nonlinear function
    # --! parameters from the raw timeseries. In constrast to
    # --! static kernels in convolutional neural networks,
    # --! the kernels here are dynamic, because they
    # --! are produced from the timeseries every time.
    param_kernsize: int


class detuning_fitphase(interface):

    def __init__(self, model):
        super().__init__()

        self.model = model

    @abstractmethod
    def enter(self):
        return

    @abstractmethod
    def run(self, timeseries):
        return

    @abstractmethod
    def next(self) -> bool:
        return


class detuning_nofit(detuning_fitphase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self):
        return

    def run(self, timeseries):
        return None

    def next(self):
        self.model._set_fitphase(self.model._get_fitphase_sta_mean())
        self.model._get_fitphase().enter()
        return True


class detuning_stafit_mean(detuning_fitphase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self):
        print("inf >> training: entering stationary mean phase")

        self.model.operator_dyn.freeze_mean()

        self.model.operator_sta.unfreeze()
        self.model.operator_sta.freeze_var()

    def run(self, timeseries):
        o = self.model.forward(timeseries, 1.0)

        timeseries_predict           = o[0]
        sta_timeseries_predict_mean  = o[1]
        sta_timeseries_predict_var   = o[2]
        dyn_timeseries_predict       = o[3]
        sta_fun                      = o[4]
        sta_fun_predict              = o[5]
        dyn_fun                      = o[6]
        dyn_fun_predict              = o[7]

        loss_dmd = self.model._fit_prediction(timeseries, sta_timeseries_predict_mean)
        loss_lin = self.model._fit_linearity(sta_fun, sta_fun_predict)

        loss = loss_dmd + loss_lin

        return loss, loss_dmd, loss_lin, 0.

    def next(self):
        self.model._set_fitphase(self.model._get_fitphase_sta_var())
        self.model._get_fitphase().enter()
        return True

class detuning_stafit_var(detuning_fitphase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self):
        print("inf >> training: entering stationary variance phase")

        self.model.operator_dyn.freeze_mean()

        self.model.operator_sta.unfreeze()
        self.model.operator_sta.freeze_mean()

    def run(self, timeseries):
        o = self.model.forward(timeseries, 1.0)

        timeseries_predict           = o[0]
        sta_timeseries_predict_mean  = o[1]
        sta_timeseries_predict_var   = o[2]
        dyn_timeseries_predict       = o[3]
        sta_fun                      = o[4]
        sta_fun_predict              = o[5]
        dyn_fun                      = o[6]
        dyn_fun_predict              = o[7]

        loss_dmd = self.model._fit_sta(timeseries, sta_timeseries_predict_mean, sta_timeseries_predict_var)

        loss = loss_dmd
        return loss, loss_dmd, 0., 0.

    def next(self):
        self.model._set_fitphase(self.model._get_fitphase_dyn())
        self.model._get_fitphase().enter()
        return True


class detuning_dynfit(detuning_fitphase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self):
        print("inf >> training: entering dynamic phase")
        self.model.operator_dyn.unfreeze()

        self.model.operator_sta.freeze_mean()
        self.model.operator_sta.freeze_var()

    def run(self, timeseries):
        o = self.model.forward(timeseries, 0.)

        timeseries_predict           = o[0]
        sta_timeseries_predict_mean  = o[1]
        sta_timeseries_predict_var   = o[2]
        dyn_timeseries_predict       = o[3]
        sta_fun                      = o[4]
        sta_fun_predict              = o[5]
        dyn_fun                      = o[6]
        dyn_fun_predict              = o[7]

        loss_transformer = self.model._fit_prediction(timeseries, dyn_timeseries_predict)
        loss_lin = self.model._fit_linearity(dyn_fun, dyn_fun_predict)

        loss = loss_transformer + loss_lin

        return loss, loss_transformer, 0., loss_lin

    def next(self):
        return False


class detuning(torch.nn.Module):
    """
    Models detuning of a cavity resonance as predictive one-dimensional timeseries.
    """

    def __init__(self, config):
        super().__init__()

        self.timeseries_ndim       = config.timeseries_ndim
        self.timeseries_nsample    = config.timeseries_nsample
        self.timestep              = config.timestep
        self.param_kernsize        = config.param_kernsize

        self.operator_sta = operator_sta(config)
        self.operator_dyn = operator_dyn(config)

        self._fitphase_sta_mean = detuning_stafit_mean(self)
        self._fitphase_sta_var  = detuning_stafit_var(self)
        self._fitphase_dyn      = detuning_dynfit(self)
        self._fitphase          = detuning_nofit(self)

    def fit_next(self):
        return self._fitphase.next()

    def fit(self, timeseries):
        """Fits internal neural networks to predict given ``timeseries``."""

        return self._fitphase.run(timeseries)

    def forward(self, timeseries, alpha):
        """Predicts given ``timeseries``."""

        if timeseries.shape[1] % self.param_kernsize:
            raise Exception('the size of timeseries is not a multiple of a kernel size')

        sta_timeseries_predict_mean, sta_timeseries_predict_var, sta_fun, sta_fun_predict = self.operator_sta(timeseries)
        dyn_timeseries_predict, dyn_fun, dyn_fun_predict = self.operator_dyn(timeseries)

        # --! blend predicted timeseries using alpha
        timeseries_predict = alpha * sta_timeseries_predict_mean + (1 - alpha) * dyn_timeseries_predict

        o = (
            timeseries_predict,
            sta_timeseries_predict_mean, sta_timeseries_predict_var,
            dyn_timeseries_predict,
            sta_fun, sta_fun_predict,
            dyn_fun, dyn_fun_predict
        )

        return o

    def _fit_sta(self, timeseries, timeseries_predict_mean, timeseries_predict_var):
        logvar = torch.clamp(timeseries_predict_var, min=-5, max=5)
        var    = torch.exp(logvar) + 1e-6

        loss_fun = torch.nn.GaussianNLLLoss()
        return loss_fun(timeseries_predict_mean, timeseries, var)

    def _fit_prediction(self, timeseries, timeseries_pred):

        loss_fn = torch.nn.MSELoss(reduction='mean')
        return loss_fn(timeseries_pred, timeseries)

    def _fit_linearity(self, fun, fun_predict):

        loss_fun = torch.nn.MSELoss(reduction='mean')
        return loss_fun(fun_predict, fun)

    def _get_fitphase_sta_mean(self):
        return self._fitphase_sta_mean

    def _get_fitphase_sta_var(self):
        return self._fitphase_sta_var

    def _get_fitphase_dyn(self):
        return self._fitphase_dyn

    def _get_fitphase(self):
        return self._fitphase

    def _set_fitphase(self, phase):
        self._fitphase = phase


@dataclass
class alpha_fun_config:
    kern_sz: int


class alpha_fun(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.kern_sz = config.kern_sz

        # --! based on kernel size, compute the size of padding
        pad_sz = (self.kern_sz - 1) // 2

        self.net = torch.nn.Sequential(

            # --! Derives 16 features from input timeseries by looking at small chunks of
            # --! length 5 using a kernel. When sliding along the timeseries,
            # --! the kernel shifts by 1 sample (stride=1). And to make
            # --! the kernel fully processes the last sample, the
            # --! timeseries are padded by two zero samples (padding=2).
            #
            # --! derived features of length 200 are downsampled by a factor of 2, see MaxPool1d
            torch.nn.Conv1d(1, 16, kernel_size=self.kern_sz, stride=1, padding=pad_sz),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            # --! extracts 32 high-level features from the already derived 16 features
            #
            # --! pools across the entire time axis and converts vector-features into scalar-features,
            # --! see AdaptiveAvgPool1d
            torch.nn.Conv1d(16, 32, kernel_size=self.kern_sz, stride=1, padding=pad_sz),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),

            # --! classifies features into a probability that given time series are stationary
            torch.nn.Flatten(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, timeseries):
        """
        Forwards ``timeseries`` to the underlying convolutional neural network (CNN). The ``timeseries``
        are expected to have a shape of [B, T, C], where B, T and C are the number of
        batch elements, timesamples and data channels, respectively.
        """
        # --! when calling this network we transpose input timeseries to shape them as [C, T],
        # --! because a convolutional neural network expects the number of
        # --! time sample T to be the last dimension
        return self.net(timeseries.transpose(-1, -2))
