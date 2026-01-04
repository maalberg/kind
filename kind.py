# --! Implementation of Kalman-inspired neural decomposition, or KIND --!

from abc import abstractmethod
from abc import ABC as interface

import torch
import numpy as np
import argparse
import time
import json

import util_data
import util_nn


def create_args_parser():
    """ Creates a command-line parser for KIND arguments. """
    parser = argparse.ArgumentParser(description='KIND timeseries forecasting')

    # --! data arguments
    parser.add_argument('--file_dir', type=str, required=True, help='relative path to file directory')
    parser.add_argument('--file_name', type=str, required=True, help='file name with no extension (e.g., .csv) and no suffix (e.g., _stat)')
    parser.add_argument('--file_ext', type=str, required=False, default='.csv', help='data file extension')
    parser.add_argument('--data_nsample', type=int, required=True, help='number of samples in timeseries stored in data')
    parser.add_argument('--data_train_size', type=float, required=False, default=0.8, help='dataset part to include in training')
    parser.add_argument('--data_test_size', type=float, required=False, default=0.5, help='non-train part to include in test, rest is validation')
    parser.add_argument('--feature_ndim', type=int, required=True, help='number of feature dimensions in data')
    parser.add_argument('--target_ndim', type=int, required=True, help='number of target dimensions in data')

    # --! forecasting arguments
    parser.add_argument('--lookback_nsample', type=int, required=True, help='number of samples in a lookback window')
    parser.add_argument('--forecast_nsample', type=int, required=True, help='number of samples in a forecast window')

    # --! training
    parser.add_argument('--batch_size', type=int, required=False, default=128, help='training batch size')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001, help='learning rate during training')
    parser.add_argument('--weight_decay', type=float, required=False, default=0.0001, help='weight decay to regularize training')
    parser.add_argument('--nepoch', type=int, required=False, default=100, help='number of training epochs')
    parser.add_argument('--patience', type=int, required=False, default=10, help='patience for early stopping during training')
    parser.add_argument('--checkpoints', type=str, required=True, help='location of model training checkpoints')

    # --! KIND
    parser.add_argument('--seg_nsample_stat', type=int, required=True, help='number of samples in a stationary data slice')
    parser.add_argument('--seg_nsample_trans', type=int, required=True, help='number of samples in a transient data slice')
    parser.add_argument('--fun_stat', type=json.loads, required=True, help='embedded functions for stationary operator')
    parser.add_argument('--fun_trans', type=json.loads, required=True, help='embedded functions for transient operator')
    parser.add_argument('--nneuron_stat', type=int, required=False, default=64, help='number of neurons in stationary operator layers')
    parser.add_argument('--nlayer_stat', type=int, required=False, default=3, help='number of layers in stationary operator')
    parser.add_argument('--nneuron_trans', type=int, required=False, default=64, help='number of neurons in transient operator layers')
    parser.add_argument('--nlayer_trans', type=int, required=False, default=3, help='number of layers in transient operator')

    return parser


class model(torch.nn.Module):
    """ Models Kalman-inpired neural decomposition, or KIND.

    This model captures the evolution of timeseries by first decomposing them into stationary and
    transient components, forecasting these components into the future and
    finally blending the forecasts in a Kalman-inspired manner.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model_nom = operator_nom(args)
        self.model_exc = operator_exc(args)

    def forward(self, lookback):

        # --! execute both operators on the given lookback
        mean_nom, zeta_nom, fun_nom, fun_stat_pred, dfun_stat, dfun_stat_pred = self.model_nom(lookback)
        mean_exc, zeta_exc, fun_exc, fun_trans_pred, dfun_trans, dfun_trans_pred = self.model_exc(lookback)

        # --! derive alpha
        alpha = zeta_exc / (zeta_exc + zeta_nom)

        # --! blend the two types of time series using the derived alpha to get the final prediction
        prediction = alpha * mean_nom + (1 - alpha) * mean_exc

        model_output = (
            prediction,
            mean_nom, zeta_nom,
            mean_exc, zeta_exc,
            fun_nom, fun_stat_pred,
            fun_exc, fun_trans_pred,
            alpha,
            dfun_stat, dfun_stat_pred,
            dfun_trans, dfun_trans_pred
        )

        return model_output


class adapter(interface):

    @abstractmethod
    def forward(self, lookback):
        return

    @abstractmethod
    def train(self, mode=True):
        return

    @abstractmethod
    def eval(self):
        return

    @property
    @abstractmethod
    def args(self):
        return

    @property
    @abstractmethod
    def model_nom(self):
        return

    @property
    @abstractmethod
    def model_exc(self):
        return

    @abstractmethod
    def parameters(self):
        return

    @abstractmethod
    def state_dict(self):
        return

    @abstractmethod
    def load_state_dict(self, state_dict):
        return

    def __call__(self, lookback):
        return self.forward(lookback)


class training:
    """Manages a phase-by-phase training of a KIND model."""

    def __init__(self, model):

        self.model = model

        # --! create internal states that denote training phases
        self.phase_mean_nom = mean_training_nom(self)
        self.phase_zeta_nom = zeta_training_nom(self)
        self.phase_mean_exc = mean_training_exc(self)
        self.phase_zeta_exc = zeta_training_exc(self)

        # --! we start with the training of nominal mean
        self.phase = self.phase_mean_nom

    def fit_next(self):
        return self.get_phase().next()

    def fit(self, dataset):

        args = self.model.args

        # --! make model initializations, data loading, optimizer selection, etc.
        #
        # --! select optimizer after initializing this model!
        self.get_phase().init_model()
        train_loader, valid_loader, test_loader = self.get_phase().load_data(dataset)
        model_optim = self.select_optimizer()
        early_stopping = util_nn.early_stopping(patience=args.patience, checkpoint_path=args.checkpoints)

        # --! start training
        for epoch in range(args.nepoch):
            train_loss = []

            for back, fore in train_loader:

                # --! since we compute a full reconstruction loss at the moment, concatenate current
                # --! lookback and forecast windows to get the full timeseries
                # --! to serve as the truth
                truth = torch.cat([back, fore], dim=1)

                # --! extract target dimensions
                truth = dataset.extract_target(truth)

                model_optim.zero_grad()

                # --! forward pass
                loss = self.get_phase().compute_loss(truth, self.get_phase().forward(back))
                train_loss.append(loss.item())

                # --! backward pass
                loss.backward()

                # --! finalize this training iteration
                model_optim.step()

            train_loss = np.average(train_loss)
            valid_loss = self.validate(dataset, valid_loader)
            test_loss = self.validate(dataset, test_loader)

            print(f'\tepoch {epoch+1} losses: train={train_loss:.6f}, valid={valid_loss:.6f}, test={test_loss:.6f}')

            # --! use validation loss to check early stopping
            if early_stopping(self.model, valid_loss):
                print("\tearly stopping ...")
                break

            # --! adjust learning rate here

        best_model_path = args.checkpoints + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        return

    def get_phase_mean_nom(self):
        return self.phase_mean_nom

    def get_phase_zeta_nom(self):
        return self.phase_zeta_nom

    def get_phase_mean_exc(self):
        return self.phase_mean_exc

    def get_phase_zeta_exc(self):
        return self.phase_zeta_exc

    def get_phase(self):
        return self.phase

    def set_phase(self, phase):
        self.phase = phase

    def validate(self, dataset, data_loader):

        total_loss = []

        # --! set this model into evaluation mode
        self.model.eval()

        with torch.no_grad():
            for back, fore in data_loader:

                # --! since we compute a full reconstruction loss at the moment, concatenate current
                # --! lookback and forecast windows to get the full timeseries
                # --! to serve as the truth
                truth = torch.cat([back, fore], dim=1)

                # --! extract target dimensions
                truth = dataset.extract_target(truth)

                loss = self.get_phase().compute_loss(truth, self.get_phase().forward(back))
                total_loss.append(loss)

        # --! reset this model back to training mode
        self.model.train()

        return np.average(total_loss)

    def select_optimizer(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.model.args.learning_rate,
            weight_decay=self.model.args.weight_decay)


class training_phase(interface):
    """Describes a KIND training phase."""

    def __init__(self, training):
        super().__init__()

        # --! reference to the training procedure
        self.training = training

    @abstractmethod
    def init_model(self):
        """ Initializes this KIND model before training. """
        return

    @abstractmethod
    def load_data(self, dataset):
        """ Loads training, validation and test data from a given ``dataset``. """
        return

    @abstractmethod
    def forward(self, lookback):
        """ Executes a forward pass of a KIND model. """
        return

    @abstractmethod
    def compute_loss(self, true, pred):
        """ Computes loss based on ``true`` and ``pred``icted timeseries. """
        return

    @abstractmethod
    def next(self) -> bool:
        """ Transitions to next state, or immediately returns False if there is no next state. """
        return

    def apply_criterion_mean(self, timeseries, timeseries_pred):
        criterion = torch.nn.MSELoss(reduction='mean')
        return criterion(timeseries_pred, timeseries)

    def apply_criterion_uncertain(self, timeseries, timeseries_pred_mean, timeseries_pred_uncertain):
        # --! clamp a log-variance to avoid big numbers
        uncertainty_log = torch.clamp(timeseries_pred_uncertain, min=-10, max=5)

        # --! convert the log variance into variance
        uncertainty = torch.exp(-uncertainty_log)

        # --! compute a negative log-likelihood manually, instead of calling torch.nn.GaussianNLLLoss
        loss = 0.5 * (uncertainty_log + (timeseries_pred_mean - timeseries)**2 * uncertainty)
        return loss.mean()


class mean_training_nom(training_phase):

    def __init__(self, training):
        super().__init__(training)

    def init_model(self):

        print('>>> training nominal mean >>>')

        model = self.training.model

        model.model_exc.freeze_mean()
        model.model_exc.freeze_var()

        model.model_nom.unfreeze()
        model.model_nom.freeze_var()

    def load_data(self, dataset):
        return dataset.load(data_type='nom')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_pred = pred[1] # < stationary prediction

        fun = pred[5]
        fun_pred = pred[6]

        loss_recon = self.apply_criterion_mean(timeseries, timeseries_pred)
        loss_linear = self.apply_criterion_mean(fun, fun_pred)
        loss = loss_recon + loss_linear

        return loss

    def next(self):
        self.training.set_phase(self.training.get_phase_zeta_nom())
        return True


class zeta_training_nom(training_phase):

    def __init__(self, training):
        super().__init__(training)

    def init_model(self):

        print('>>> training nominal uncertainty >>>')

        model = self.training.model

        model.model_exc.freeze_mean()
        model.model_exc.freeze_var()

        model.model_nom.unfreeze()
        model.model_nom.freeze_mean()

    def load_data(self, dataset):
        return dataset.load(data_type='mixed')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_pred_mean = pred[1]
        timeseries_u_pre = pred[2]
        fun_u = pred[10]
        fun_u_pre = pred[11]

        loss_linear = self.apply_criterion_mean(fun_u, fun_u_pre)
        loss_uncertain = self.apply_criterion_uncertain_test(timeseries, timeseries_pred_mean, timeseries_u_pre)
        loss = loss_uncertain + loss_linear

        return loss

    def next(self):
        self.training.set_phase(self.training.get_phase_mean_exc())
        return True

    def apply_criterion_uncertain_test(self, true, pre, pre_u):
        err = torch.abs(true - pre)
        return self.apply_criterion_mean(err, pre_u)


class mean_training_exc(training_phase):

    def __init__(self, training):
        super().__init__(training)

    def init_model(self):

        print('>>> training excursion mean >>>')

        model = self.training.model

        model.model_exc.unfreeze()
        model.model_exc.freeze_var()

        model.model_nom.freeze_mean()
        model.model_nom.freeze_var()

    def load_data(self, dataset):
        return dataset.load(data_type='exc')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_pred_mean = pred[3]

        fun = pred[7]
        fun_pred = pred[8]

        loss_recon = self.apply_criterion_mean(timeseries, timeseries_pred_mean)
        loss_linear = self.apply_criterion_mean(fun, fun_pred)
        loss = loss_recon + loss_linear

        return loss

    def next(self):
        self.training.set_phase(self.training.get_phase_zeta_exc())
        return True


class zeta_training_exc(training_phase):

    def __init__(self, training):
        super().__init__(training)

    def init_model(self):

        print('>>> training excursion uncertainty >>>')

        model = self.training.model

        model.model_exc.unfreeze()
        model.model_exc.freeze_mean()

        model.model_nom.freeze_mean()
        model.model_nom.freeze_var()

    def load_data(self, dataset):
        return dataset.load(data_type='mixed')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_pred_mean = pred[3]
        timeseries_u_pre = pred[4]
        fun_u = pred[12]
        fun_u_pre = pred[13]

        loss_linear = self.apply_criterion_mean(fun_u, fun_u_pre)
        loss_uncertain = self.apply_criterion_uncertain_test(timeseries, timeseries_pred_mean, timeseries_u_pre)
        loss = loss_uncertain + loss_linear

        return loss

    def next(self):
        return False

    def apply_criterion_uncertain_test(self, true, pre, pre_u):
        err = torch.abs(true - pre)
        return self.apply_criterion_mean(err, pre_u)
    

class operator(torch.nn.Module, interface):
    """Models timeseries dynamics."""

    sin_nparam  = 2
    cos_nparam  = 2
    data_nparam = 1
    poly_nparam = 1
    exp_nparam  = 1

    def __init__(self, args):
        super().__init__()

        # --! store mutual configuration inside this base class
        self.nfeature = args.feature_ndim
        self.ntarget = args.target_ndim
        self.back_nsample = args.lookback_nsample
        self.fore_nsample = args.forecast_nsample

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

    def _eval_sinx(self, param):
        amp, ang = torch.split(param, 1, dim=-1)
        return amp * torch.sin(ang)

    def _eval_sin(self, param):
        param = torch.split(param, self.sin_nparam, dim=-1)
        return torch.cat([self._eval_sinx(p) for p in param], dim=-1)

    def _eval_cosx(self, param):
        amp, ang = torch.split(param, 1, dim=-1)
        return amp * torch.cos(ang)

    def _eval_cos(self, param):
        param = torch.split(param, self.cos_nparam, dim=-1)
        return torch.cat([self._eval_cosx(p) for p in param], dim=-1)

    def _eval_data(self, param):
        return param

    def _eval_poly(self, param):
        param = torch.split(param, self.poly_nparam, dim=-1)
        degree = torch.arange(len(param)) + 1
        return torch.cat([p**d for p, d in zip(param, degree)], dim=-1)

    def _eval_exp(self, param):
        return torch.exp(param)


class operator_nom(operator):
    """ Models dynamics of nominal timeseries in a DMD-like manner. """

    def __init__(self, args):

        # --! initialize common operator parameters
        super().__init__(args)

        # --! this here is a convenient place to get the total number of functions, because
        # --! later the function configuration becomes respecified
        # --! to facilitate other things
        nfun = sum(args.fun_stat.values())

        # --! initialize stationary-specific parameters
        #
        # --! note that here the function configuration becomes respecified
        self.fun            = self._respec_fun(args.fun_stat)
        self.param_kernsize = args.seg_nsample_stat

        if self.fore_nsample % self.param_kernsize:
            raise Exception('the number of forecast samples must be a multiple of a kernel size!')
        self.fun_nsample_forecast  = self.fore_nsample // self.param_kernsize

        # --! here the function configuration is already respecified, such that it is convenient
        # --! to get the total number of required function parameters
        nparam      = sum(self.fun.values())
        fun_nsample = self.back_nsample // self.param_kernsize

        # --! create an encoder to encode timeseries into embedded functions
        #
        # --! more precisely, input timeseries are partitioned into slices, and the encoder
        # --! encodes slice-specific kernels for every function parameter
        # --! in the embedded (latent) space
        fun_enc_ni   = self.param_kernsize * self.nfeature
        fun_enc_no   = nparam * self.param_kernsize * self.nfeature
        fun_enc_feat = util_nn.make_feat(ni=fun_enc_ni, no=fun_enc_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.fun_enc = util_nn.fcnn(feat=fun_enc_feat, actfun_hid='relu')
        self.fun_u_enc = util_nn.fcnn(feat=fun_enc_feat, actfun_hid='relu')

        if self.nfeature > 1:
            # --! this linear transformation is supposed to prune the dimensionality of the
            # --! basis functions, such that only the number of these basis functions
            # --! influences the order of the DMD matrix, whereas the number of data
            # --! dimensions has no effect on the order
            fun_prune_ni = nfun * self.nfeature
            fun_prune_no = nfun
            self.fun_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)
            self.fun_u_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)

        # --! create a DMD-like model (a matrix) that captures stationary (mean) dynamics
        #
        # --! since this matrix is learned only once and does not adapt during runtime,
        # --! this operator can also be called static, instead of stationary
        mod_mean_ni = nfun
        mod_mean_no = nfun
        self.mod_mean = torch.nn.Linear(mod_mean_ni, mod_mean_no, bias=False)

        # --! create a generator that produces models capturing the evolution of variance (uncertainty)
        #
        # --! the generator takes a flattened sequence of function values and returns
        # --! a flattened sequence of square matrices
        mod_var_gen_ni = fun_nsample * nfun
        mod_var_gen_no = (fun_nsample + self.fun_nsample_forecast) * nfun * nfun
        mod_var_gen_feat = util_nn.make_feat(ni=mod_var_gen_ni, no=mod_var_gen_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.mod_u_gen = util_nn.fcnn(feat=mod_var_gen_feat, actfun_hid='relu')

        # --! create prediction decoders to decode predicted embeddings back to timeseries and uncertainty
        pre_dec_ni = nfun
        pre_dec_no = self.param_kernsize * self.ntarget
        pre_dec_feat = util_nn.make_feat(ni=pre_dec_ni, no=pre_dec_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.pre_mean_dec = util_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')
        self.pre_u_dec  = util_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.nfeature)

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
            self.param_kernsize * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.nfeature)

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
        return self.fun_prune(fun) if self.nfeature > 1 else fun

    def embed_uncertainty(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.nfeature)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.fun_u_enc(i)

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
            self.param_kernsize * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.nfeature)

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
        return self.fun_u_prune(fun) if self.nfeature > 1 else fun

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

        # --! constants for convenience
        batsize     = errors.shape[0]
        err_nsample = errors.shape[1]
        nfun        = errors.shape[2]

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of error values
        i           = torch.flatten(errors, start_dim=1)
        mat         = self.mod_var_gen(i).reshape(batsize, -1, nfun, nfun)
        mat         = util_nn.cumprod_mat(mat[:, 1:])

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

    def predict_uncertainty(self, functions):

        # --! constants for convenience
        batsize = functions.shape[0]
        fun_nsample = functions.shape[1]
        nfun = functions.shape[2]

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of function values
        i = torch.flatten(functions, start_dim=1)
        mat = self.mod_u_gen(i).reshape(batsize, -1, nfun, nfun)
        mat = util_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial condition of function history
        #
        # --! the initial condition (ic) is shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by the matrices
        fun_ic = torch.unsqueeze(functions[:, :1], -2)

        # --! predict the evolution of functions by multiplying their initial conditions by the matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun],
        # --! i.e. a batch with function trajectories consisting of individual error-points [1, nfun]
        fun_pre = torch.matmul(fun_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        fun_pre = torch.squeeze(fun_pre, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(fun_pre, [fun_nsample - 1, self.fun_nsample_forecast], dim=1)

    def forward(self, timeseries):

        # --! from given timeseries, embed latent function values and then predict these embeddings
        # --! starting from the first value (initial condition) upto a specified horizon
        fun = self.embed(timeseries)
        fun_pre, fun_pre_forecast  = self.predict_mean(fun)
        fun_pre = torch.cat([fun[:, :1], fun_pre], dim=1)

        fun_u = self.embed_uncertainty(timeseries)
        fun_u_pre, fun_u_pre_forecast = self.predict_uncertainty(fun_u)
        fun_u_pre = torch.cat([fun_u[:, :1], fun_u_pre], dim=1)

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        timeseries_pre_mean  = self.pre_mean_dec(torch.cat([fun_pre, fun_pre_forecast], dim=1))
        timeseries_pre_mean  = timeseries_pre_mean.reshape(timeseries_pre_mean.shape[0], -1, self.ntarget)

        # --! decode predicted and denormalized function uncertainty to model uncertainty
        timeseries_u_pre = self.pre_u_dec(torch.cat([fun_u_pre, fun_u_pre_forecast], dim=1))
        timeseries_u_pre = timeseries_u_pre.reshape(timeseries_u_pre.shape[0], -1, self.ntarget)

        return timeseries_pre_mean, timeseries_u_pre, fun, fun_pre, fun_u, fun_u_pre

    def freeze_mean(self):
        util_nn.freeze_module(self.fun_enc)
        if self.nfeature > 1:
            util_nn.freeze_module(self.fun_prune)
        util_nn.freeze_module(self.mod_mean)
        util_nn.freeze_module(self.pre_mean_dec)

    def freeze_var(self):
        util_nn.freeze_module(self.fun_u_enc)
        if self.nfeature > 1:
            util_nn.freeze_module(self.fun_u_prune)
        util_nn.freeze_module(self.mod_u_gen)
        util_nn.freeze_module(self.pre_u_dec)

    def unfreeze(self):
        util_nn.unfreeze_module(self.fun_enc)
        util_nn.unfreeze_module(self.fun_u_enc)
        if self.nfeature > 1:
            util_nn.unfreeze_module(self.fun_prune)
            util_nn.unfreeze_module(self.fun_u_prune)
        util_nn.unfreeze_module(self.mod_mean)
        util_nn.unfreeze_module(self.mod_u_gen)
        util_nn.unfreeze_module(self.pre_mean_dec)
        util_nn.unfreeze_module(self.pre_u_dec)


class operator_exc(operator):
    """Models dynamics of excursion timeseries using a Transformer-based attention mechanism."""

    def __init__(self, args):

        # --! initialize common operator parameters
        super().__init__(args)

        # --! this here is a convenient place to get the total number of functions, because
        # --! later the function configuration becomes respecified
        # --! to facilitate other things
        nfun = sum(args.fun_trans.values())

        # --! initialize transient-specific parameters
        #
        # --! note that here the function configuration becomes respecified
        self.fun            = self._respec_fun(args.fun_trans)
        self.param_kernsize = args.seg_nsample_trans

        if self.fore_nsample % self.param_kernsize:
            raise Exception('the number of forecast samples must be a multiple of a kernel size!')
        self.fun_nsample_forecast  = self.fore_nsample // self.param_kernsize

        # --! here the function configuration is already respecified, such that it is convenient
        # --! to get the total number of required function parameters
        nparam = sum(self.fun.values())
        fun_nsample = self.back_nsample // self.param_kernsize

        # --! create an MLP-based encoder to encode timeseries into embedded functions
        #
        # --! more precisely, input timeseries are partitioned into slices, and the encoder
        # --! encodes slice-specific kernels for every function parameter
        # --! in the embedded (latent) space
        fun_enc_ni   = self.param_kernsize * self.nfeature
        fun_enc_no   = nparam * self.param_kernsize * self.nfeature
        fun_enc_feat = util_nn.make_feat(ni=fun_enc_ni, no=fun_enc_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.fun_enc = util_nn.fcnn(feat=fun_enc_feat, actfun_hid='relu')
        self.fun_u_enc = util_nn.fcnn(feat=fun_enc_feat, actfun_hid='relu')

        if self.nfeature > 1:
            # --! this linear transformation is supposed to prune the dimensionality of the
            # --! basis functions, such that only the number of these basis functions
            # --! influences the order of linear matrices and the number of data
            # --! dimensions has no effect on that order
            fun_prune_ni = nfun * self.nfeature
            fun_prune_no = nfun
            self.fun_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)
            self.fun_u_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)

        # --! encoder network which learns to attend over slices of embedded function values
        mod_mean_att_enc_ni = nfun

        # --! the attention encoder is implemented in terms of a Transformer encoder network
        self.mod_mean_att_enc = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=mod_mean_att_enc_ni,
                nhead=1,
                dim_feedforward=args.nneuron_trans,
                batch_first=True),
            num_layers=args.nlayer_trans,
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
        mod_mean_gen_feat = util_nn.make_feat(ni=mod_mean_gen_ni, no=mod_mean_gen_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.mod_mean_gen = util_nn.fcnn(feat=mod_mean_gen_feat, actfun_hid='relu')

        # --! create a generator that produces models capturing the evolution of variance (uncertainty)
        #
        # --! the generator takes a flattened sequence of function values and returns
        # --! a flattened sequence of square matrices
        mod_u_gen_ni = fun_nsample * nfun
        mod_u_gen_no = (fun_nsample + self.fun_nsample_forecast) * nfun * nfun
        mod_u_gen_feat = util_nn.make_feat(ni=mod_u_gen_ni, no=mod_u_gen_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.mod_u_gen = util_nn.fcnn(feat=mod_u_gen_feat, actfun_hid='relu')

        # --! create MLP-based decoders to decode embeddings back to timeseries with uncertainty
        pre_dec_ni = nfun
        pre_dec_no = self.param_kernsize * self.ntarget
        pre_dec_feat = util_nn.make_feat(ni=pre_dec_ni, no=pre_dec_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.pre_mean_dec = util_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')
        self.pre_u_dec = util_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.nfeature)

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
            self.param_kernsize * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.nfeature)

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
        return self.fun_prune(fun) if self.nfeature > 1 else fun

    def embed_uncertainty(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * self.nfeature)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.fun_u_enc(i)

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
            self.param_kernsize * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.nfeature)

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
        return self.fun_u_prune(fun) if self.nfeature > 1 else fun

    def predict_mean(self, functions):

        batsize     = functions.shape[0]
        fun_nsample = functions.shape[1]
        nfun        = functions.shape[2]

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
        mat = util_nn.cumprod_mat(mat[:, 1:])

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

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of error values
        i           = torch.flatten(errors, start_dim=1)
        mat         = self.mod_var_gen(i).reshape(batsize, -1, nfun, nfun)
        mat         = util_nn.cumprod_mat(mat[:, 1:])

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

    def predict_uncertainty(self, functions):

        # --! constants for convenience
        batsize = functions.shape[0]
        fun_nsample = functions.shape[1]
        nfun = functions.shape[2]

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of function values
        i = torch.flatten(functions, start_dim=1)
        mat = self.mod_u_gen(i).reshape(batsize, -1, nfun, nfun)
        mat = util_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial condition of function history
        #
        # --! the initial condition (ic) is shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by the matrices
        fun_ic = torch.unsqueeze(functions[:, :1], -2)

        # --! predict the evolution of functions by multiplying their initial conditions by the matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun],
        # --! i.e. a batch with function trajectories consisting of individual error-points [1, nfun]
        fun_pre = torch.matmul(fun_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        fun_pre = torch.squeeze(fun_pre, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(fun_pre, [fun_nsample - 1, self.fun_nsample_forecast], dim=1)

    def forward(self, timeseries):

        # --! from given timeseries, embed latent function values and then predict these embeddings
        # --! starting from the first value (initial condition) upto a specified horizon
        fun                       = self.embed(timeseries)
        fun_pre, fun_pre_forecast = self.predict_mean(fun)
        fun_pre                   = torch.cat([fun[:, :1], fun_pre], dim=1)

        fun_u = self.embed_uncertainty(timeseries)
        fun_u_pre, fun_u_pre_forecast = self.predict_uncertainty(fun_u)
        fun_u_pre = torch.cat([fun_u[:, :1], fun_u_pre], dim=1)

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        timeseries_pre_mean = self.pre_mean_dec(torch.cat([fun_pre, fun_pre_forecast], dim=1))
        timeseries_pre_mean = timeseries_pre_mean.reshape(timeseries_pre_mean.shape[0], -1, self.ntarget)

        # --! decode predicted and denormalized function uncertainty to model uncertainty
        timeseries_u_pre = self.pre_u_dec(torch.cat([fun_u_pre, fun_u_pre_forecast], dim=1))
        timeseries_u_pre = timeseries_u_pre.reshape(timeseries_u_pre.shape[0], -1, self.ntarget)

        return timeseries_pre_mean, timeseries_u_pre, fun, fun_pre, fun_u, fun_u_pre

    def freeze_mean(self):
        util_nn.freeze_module(self.fun_enc)
        if self.nfeature > 1:
            util_nn.freeze_module(self.fun_prune)
        util_nn.freeze_module(self.mod_mean_att_enc)
        util_nn.freeze_module(self.mod_mean_gen)
        util_nn.freeze_module(self.pre_mean_dec)

    def freeze_var(self):
        util_nn.freeze_module(self.fun_u_enc)
        if self.nfeature > 1:
            util_nn.freeze_module(self.fun_u_prune)
        util_nn.freeze_module(self.mod_u_gen)
        util_nn.freeze_module(self.pre_u_dec)

    def unfreeze(self):
        util_nn.unfreeze_module(self.fun_enc)
        util_nn.unfreeze_module(self.fun_u_enc)
        if self.nfeature > 1:
            util_nn.unfreeze_module(self.fun_prune)
            util_nn.unfreeze_module(self.fun_u_prune)
        util_nn.unfreeze_module(self.mod_mean_att_enc)
        util_nn.unfreeze_module(self.mod_mean_gen)
        util_nn.unfreeze_module(self.mod_u_gen)
        util_nn.unfreeze_module(self.pre_mean_dec)
        util_nn.unfreeze_module(self.pre_u_dec)

