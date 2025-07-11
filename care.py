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
        """Makes all trainable submodules trainable again."""
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
            deg = utils_nn.extract_poly_deg(fun)
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


class operator_stationary(operator):
    """Models dynamics of stationary timeseries in a DMD-like manner."""

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
        self.enc = utils_nn.fcnn(feat=[enc_ni, 64, 64, enc_no], actfun_hid='relu')

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
        self.dec_mean   = utils_nn.fcnn(feat=[dec_ni, 64, 64, dec_no], actfun_hid='relu')
        self.dec_var    = utils_nn.fcnn(feat=[dec_ni * 2, 64, 64, dec_no], actfun_hid='relu')

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
        utils_nn.freeze_module(self.enc)
        utils_nn.freeze_module(self.model)
        utils_nn.freeze_module(self.dec_mean)

    def freeze_var(self):
        utils_nn.freeze_module(self.dec_var)

    def unfreeze(self):
        utils_nn.unfreeze_module(self.enc)
        utils_nn.unfreeze_module(self.model)
        utils_nn.unfreeze_module(self.dec_mean)
        utils_nn.unfreeze_module(self.dec_var)


class operator_transient(operator):
    """Models dynamics of transient timeseries using a Transformer-based attention mechanism."""

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
        self.embed_enc = utils_nn.fcnn(feat=[embed_enc_ni, 64, 64, embed_enc_no], actfun_hid='relu')

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
        self.model_from = utils_nn.fcnn(feat=[nfun, 64, 64, nfun*nfun], actfun_hid='relu')

        # --! create MLP-based decoders to decode embeddings back to timeseries with uncertainty (mean and variance)
        #
        # --! features of the variance decoder are doubled to accommodate prediction errors
        dec_ni        = nfun * self.timeseries_ndim
        dec_no        = self.param_kernsize * self.timeseries_ndim
        self.dec_mean = utils_nn.fcnn(feat=[dec_ni, 64, 64, dec_no], actfun_hid='relu')
        self.dec_var  = utils_nn.fcnn(feat=[dec_ni * 3, 64, 64, dec_no], actfun_hid='relu')

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
        timeseries_predict_mean = self.dec_mean(fun_predict)
        timeseries_predict_mean = timeseries_predict_mean.reshape(timeseries_predict_mean.shape[0], -1, self.timeseries_ndim)

        # --! to facilitate variance learning we explicitly provide prediction error along the prediction itself
        dfun                     = fun_predict - fun
        dec_var_i                = torch.cat([fun, fun_predict, dfun], dim=-1)
        timeseries_predict_var   = self.dec_var(dec_var_i)
        timeseries_predict_var   = timeseries_predict_var.reshape(timeseries_predict_var.shape[0], -1, self.timeseries_ndim)

        return timeseries_predict_mean, timeseries_predict_var, fun, fun_predict

    def freeze_mean(self):
        utils_nn.freeze_module(self.embed_enc)
        utils_nn.freeze_module(self.adapt_enc)
        utils_nn.freeze_module(self.model_from)
        utils_nn.freeze_module(self.dec_mean)

    def freeze_var(self):
        utils_nn.freeze_module(self.dec_var)

    def unfreeze(self):
        utils_nn.unfreeze_module(self.embed_enc)
        utils_nn.unfreeze_module(self.adapt_enc)
        utils_nn.unfreeze_module(self.model_from)
        utils_nn.unfreeze_module(self.dec_mean)
        utils_nn.unfreeze_module(self.dec_var)


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

        if timeseries.shape[1] % self._model.param_kernsize:
            raise Exception('the size of timeseries is not a multiple of a kernel size')

        # --! execute both operators on given time series
        timeseries_sta, timeseries_sta_logvar, fun_sta, fun_sta_pred = self._model.operator_sta(timeseries)
        timeseries_trans, timeseries_trans_logvar, fun_trans, fun_trans_pred = self._model.operator_dyn(timeseries)

        # --! derive alpha
        timeseries_sta_var   = torch.exp(timeseries_sta_logvar) + 1e-6
        timeseries_trans_var = torch.exp(timeseries_trans_logvar) + 1e-6
        alpha = timeseries_trans_var / (timeseries_trans_var + timeseries_sta_var)

        # --! blend the two types of time series using the derived alpha to get the final prediction
        timeseries_pred = alpha * timeseries_sta + (1 - alpha) * timeseries_trans

        o = (
            timeseries_pred,
            timeseries_sta, timeseries_sta_logvar,
            timeseries_trans, timeseries_trans_logvar,
            fun_sta, fun_sta_pred,
            fun_trans, fun_trans_pred,
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
        """Predicts given ``timeseries`` in evaluation mode.

        The ``timeseries`` are expected to be unscaled, so that they can be scaled before the main algorithm and
        then unscaled once results are obtained.
        """

        # --! remove mean
        mean = torch.mean(timeseries, dim=1, keepdim=True)
        timeseries = timeseries - mean

        # --! scale data using minmax to a range from -1 to 1
        scaler = utils_data.minmax_scaler(feature_range=(-1, 1))
        timeseries = scaler.fit_transform(timeseries)

        o = self._forward(timeseries)

        # --! extract to-be-unscaled timeseries from the forwarded result
        timeseries_pred         = o[0]
        timeseries_sta          = o[1]
        timeseries_trans        = o[3]

        # --! unscale resulting timeseries
        timeseries_pred       = scaler.inverse_transform(timeseries_pred)
        timeseries_pred       = timeseries_pred + mean
        timeseries_sta        = scaler.inverse_transform(timeseries_sta)
        timeseries_sta        = timeseries_sta + mean
        timeseries_trans      = scaler.inverse_transform(timeseries_trans)
        timeseries_trans      = timeseries_trans + mean

        # --! put unscaled timeseries back to the result tuple and return the tuple
        o    = list(o)
        o[0] = timeseries_pred
        o[1] = timeseries_sta
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
                    timeseries = data[0][:, :self._model._fit_phase.subtimeseries_nsample, :1]
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
        self.subtimeseries_nsample = None
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
        self.subtimeseries_nsample = param['subtimeseries_nsample']
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

        self._model.operator_dyn.freeze_mean()

        self._model.operator_sta.unfreeze()
        self._model.operator_sta.freeze_var()

        self.datadir = param['stadatadir']

    def compute_loss(self, timeseries):
        timeseries_predict_mean, timeseries_predict_var, fun, fun_predict = self._model.operator_sta(timeseries)

        loss_dmd = self._compute_meanloss(timeseries, timeseries_predict_mean)
        loss_lin = self._compute_meanloss(fun, fun_predict)

        loss = loss_dmd + loss_lin

        return loss, loss_dmd, loss_lin, 0.

    def next(self):
        self._model._set_phase(self._model._get_phase_sta_var())
        return True


class phase_stationary_var(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering stationary variance phase")

        super().enter(param)

        self._model.operator_dyn.freeze_mean()

        self._model.operator_sta.unfreeze()
        self._model.operator_sta.freeze_mean()

        self.datadir = param['mixdatadir']

    def compute_loss(self, timeseries):
        timeseries_predict_mean, timeseries_predict_var, fun, fun_predict = self._model.operator_sta(timeseries)

        loss = self._compute_varloss(timeseries, timeseries_predict_mean, timeseries_predict_var)

        return loss, loss, 0., 0.

    def next(self):
        self._model._set_phase(self._model._get_phase_dyn_mean())
        return True


class phase_transient_mean(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering dynamic mean phase")

        super().enter(param)

        self._model.operator_dyn.unfreeze()
        self._model.operator_dyn.freeze_var()

        self._model.operator_sta.freeze_mean()
        self._model.operator_sta.freeze_var()

        self.datadir = param['transdatadir']

    def compute_loss(self, timeseries):
        timeseries_predict_mean, timeseries_predict_var, fun, fun_predict = self._model.operator_dyn(timeseries)

        loss_transformer = self._compute_meanloss(timeseries, timeseries_predict_mean)
        loss_lin = self._compute_meanloss(fun, fun_predict)

        loss = 1.0 * loss_transformer + 1.0 * loss_lin

        return loss, loss_transformer, 0., loss_lin

    def next(self):
        self._model._set_phase(self._model._get_phase_dyn_var())
        return True


class phase_transient_var(fit_phase):

    def __init__(self, model):
        super().__init__(model)

    def enter(self, param):
        print("inf >> fit: entering dynamic variance phase")

        super().enter(param)

        self._model.operator_dyn.unfreeze()
        self._model.operator_dyn.freeze_mean()

        self._model.operator_sta.freeze_mean()
        self._model.operator_sta.freeze_var()

        self.datadir = param['mixdatadir']

    def compute_loss(self, timeseries):
        timeseries_predict_mean, timeseries_predict_var, fun, fun_predict = self._model.operator_dyn(timeseries)

        loss = self._compute_varloss(timeseries, timeseries_predict_mean, timeseries_predict_var)
        return loss, loss, 0., 0.

    def next(self):
        return False


@dataclass
class model_config:
    """
    Stores KIND model configuration.
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


class model(torch.nn.Module):
    """Models Kalman-inpired neural decomposition, or KIND.

    This model captures the evolution of timeseries by first decomposing them into stationary and
    transient components, predicting these components into the future and
    finally blending the predictions in a Kalman-inspired manner.
    """

    def __init__(self, config):
        super().__init__()

        self.timeseries_ndim       = config.timeseries_ndim
        self.timeseries_nsample    = config.timeseries_nsample
        self.timestep              = config.timestep
        self.param_kernsize        = config.param_kernsize

        self.operator_sta = operator_stationary(config)
        self.operator_dyn = operator_transient(config)

        self._fit_phase_sta_mean = phase_stationary_mean(self)
        self._fit_phase_sta_var  = phase_stationary_var(self)
        self._fit_phase_dyn_mean = phase_transient_mean(self)
        self._fit_phase_dyn_var  = phase_transient_var(self)
        self._fit_phase          = self._fit_phase_sta_mean

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

    def _get_phase_sta_mean(self):
        return self._fit_phase_sta_mean

    def _get_phase_sta_var(self):
        return self._fit_phase_sta_var

    def _get_phase_dyn_mean(self):
        return self._fit_phase_dyn_mean

    def _get_phase_dyn_var(self):
        return self._fit_phase_dyn_var

    def _get_phase(self):
        return self._fit_phase

    def _set_phase(self, phase):
        self._fit_phase = phase

