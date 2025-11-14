# --!-------------------------------------------------------------------!
# --! Implementation of Kalman-inspired neural decomposition, or KIND
# --!-------------------------------------------------------------------!

from abc import abstractmethod
from abc import ABC as interface

import torch
import numpy as np
import argparse
import time
import json

import utils_data
import utils_nn

from utils_data import conv_str2ints, minmax_scaler

from matplotlib import pyplot as plt


def create_args_parser():
    """ Creates a command-line parser for KIND arguments. """
    parser = argparse.ArgumentParser(description='KIND timeseries forecasting')

    # --! data arguments
    parser.add_argument('--data_name', type=str, required=True, default='SRF gun simulation', help='data name')
    parser.add_argument('--data_dir', type=str, required=True, default='../../data/delay', help='path to data directory')
    parser.add_argument('--data_file', type=str, required=True, default='srfgun', help='data file name without extension and suffix')
    parser.add_argument('--data_ext', type=str, required=False, default='.csv', help='data file extension')
    parser.add_argument('--data_nsample', type=int, required=True, default=200, help='number of samples in timeseries stored in data')
    parser.add_argument('--data_scale_min', type=float, required=True, default=-1, help='data minimum value when scaled using min-max')
    parser.add_argument('--data_scale_max', type=float, required=True, default=1, help='data maximum value when scaled using min-max')
    parser.add_argument('--data_train_size', type=float, required=True, default=0.75, help='dataset part to include in training')
    parser.add_argument('--data_test_size', type=float, required=True, default=0.5, help='non-train part to include in test, rest is validation')
    parser.add_argument('--feature_dim', type=conv_str2ints, required=True, default='0', help='list of feature dimensions in data')
    parser.add_argument('--target_dim', type=conv_str2ints, required=True, default='0', help='list of target dimensions in data')
    parser.add_argument('--mask_dim', type=conv_str2ints, required=True, default='0', help='list of mask dimensions in data')

    # --! forecasting arguments
    parser.add_argument('--lookback_nsample', type=int, required=True, default=96, help='number of samples in a lookback window')
    parser.add_argument('--forecast_nsample', type=int, required=True, default=48, help='number of samples in a forecast window')

    # --! training
    parser.add_argument('--batch_size', type=int, required=True, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, required=True, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, required=True, default=0.0001, help='weight decay to regularize training')
    parser.add_argument('--nepoch', type=int, required=True, default=10, help='number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--checkpoints', type=str, default='../../models/', help='location of model training checkpoints')

    # --! KIND
    parser.add_argument('--seg_nsample_stat', type=int, required=True, default=24, help='number of samples in a stationary data segment')
    parser.add_argument('--seg_nsample_trans', type=int, required=True, default=12, help='number of samples in a transient data segment')
    parser.add_argument('--fun_stat', type=json.loads, required=True, default='{"data": 8}', help='embedded functions for stationary operator')
    parser.add_argument('--fun_trans', type=json.loads, required=True, default='{"data": 8}', help='embedded functions for transient operator')
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

        self._fit_dataset = None

        # --! declare KIND operators
        self.stationary = operator_stationary(args)
        self.transient = operator_transient(args)

        # --! specify states, or modes, of this model that will define its behavior
        self._mode_fit = model_fit(self)
        self._mode_eval = model_eval(self)
        self._mode = self._mode_fit

    def fit_next(self):
        """ Advances this model to next fit state. Returns True if the model has been advanced, False if there is no next state. """
        return self._get_mode().fit_next()

    def fit(self, dataset):
        """ Fits this KIND model to given ``dataset`` according to the model's current mode and fit state. """
        return self._get_mode().fit(dataset)

    def forward(self, timeseries):
        """ Uses given ``timeseries`` to execute a forward pass of this model according to current mode. """
        return self._get_mode().forward(timeseries)

    def train(self, mode=True):
        """ Switches this model mode according to ``mode``: True - go to fit, False - go to evaluation mode. """

        if mode is True:
            self._get_mode().train()
        else:
            self._get_mode().eval()

        # --! according to the interface of torch.nn.Module, this method returns self
        return self

    def eval(self):
        """ Sets this model into evaluation mode. """
        return self.train(mode=False) # < delegate execution to train method

    def _get_mode_fit(self):
        return self._mode_fit

    def _get_mode_eval(self):
        return self._mode_eval

    def _get_mode(self):
        return self._mode

    def _set_mode(self, mode):
        self._mode = mode

    def _train_module(self, mode=True):
        """ Calls the super class of this PyTorch module to change the behavior of particular modules. """
        super().train(mode)

class model_mode(interface):
    """ Represents current mode of a KIND model, e.g. fit or evaluate. """

    def __init__(self, model):
        super().__init__()

        # --! keep reference to a KIND model
        self.model = model

    @abstractmethod
    def train(self):
        """ Sets this model into training mode. """
        return

    @abstractmethod
    def eval(self):
        """ Sets this model into evaluation mode. """
        return

    @abstractmethod
    def fit_next(self):
        """ Advances KIND model to next fit state. Returns True if the model has been advanced, False if there is no next state. """
        return

    @abstractmethod
    def fit(self, dataset):
        """ Fits this KIND model to given ``dataset`` according to the model's current mode and fit state. """
        return

    @abstractmethod
    def forward(self, timeseries):
        """ Executes a forward pass on given ``timeseries`` according to current mode of this KIND model. """
        return

    def _forward_scaled(self, timeseries):
        """ Uses scaled ``timeseries`` to execute a forward pass of this KIND model. """

        # --! execute both operators on given time series
        timeseries_stat, timeseries_stat_logvar, fun_stat, fun_stat_pred, dfun_stat, dfun_stat_pred = self.model.stationary(timeseries)
        timeseries_trans, timeseries_trans_logvar, fun_trans, fun_trans_pred, dfun_trans, dfun_trans_pred = self.model.transient(timeseries)

        # --! derive alpha
        timeseries_stat_var  = torch.exp(timeseries_stat_logvar) + 1e-6
        timeseries_trans_var = torch.exp(timeseries_trans_logvar) + 1e-6
        alpha = timeseries_trans_var / (timeseries_trans_var + timeseries_stat_var)

        # --! blend the two types of time series using the derived alpha to get the final prediction
        timeseries_pred = alpha * timeseries_stat + (1 - alpha) * timeseries_trans

        output = (
            timeseries_pred,
            timeseries_stat, timeseries_stat_logvar,
            timeseries_trans, timeseries_trans_logvar,
            fun_stat, fun_stat_pred,
            fun_trans, fun_trans_pred,
            alpha,
            dfun_stat, dfun_stat_pred,
            dfun_trans, dfun_trans_pred
        )

        return output


class model_eval(model_mode):
    """ Represents an evaluation mode of a KIND model. """

    def __init__(self, model):
        super().__init__(model)

    def train(self):
        """ Sets this KIND model into fit mode. """
        self.model._set_mode(self.model._get_mode_fit())
        self.model._train_module(mode=True)

    def eval(self):
        """ Returns immediately as KIND model is already in evaluation mode. """
        return
    
    def fit_next(self):
        return False

    def fit(self, dataset):
        """ Returns immediately as KIND model is currently in evaluation mode. """
        return

    def forward(self, timeseries):
        """ Uses unnormalized ``timeseries`` to execute a forward pass of KIND model in evaluation mode. """

        # --! combine feature and target dimension indeces into a unique-valued set
        #dim = self.model.args.feature_dim + self.model.args.target_dim
        #dim = set(dim)

        # --! collect means of data dimensions into dictionary, where dimension indeces serve as keys
        #mean = dict([(k, torch.mean(timeseries[:, :, [k]], dim=1, keepdim=True)) for k in dim])

        # --! create scalers for data dimensions as a dictionary, where dimension indeces serve as keys
        #scaler_range = (self.model.args.data_scale_min, self.model.args.data_scale_max)
        #scaler = dict([(k, minmax_scaler(scaler_range)) for k in dim])

        # --! scale timeseries
        #timeseries = torch.cat([timeseries[:, :, [k]] - mean[k] for k in dim], dim=-1)
        #timeseries = torch.cat([scaler[k].fit_transform(timeseries[:, :, [k]], dim=1) for k in dim], dim=-1)

        #detuning, control, mask = torch.split(timeseries, 1, dim=-1)

        #detuning_mean = torch.mean(detuning, dim=1, keepdim=True)
        #control_mean = torch.mean(control, dim=1, keepdim=True)

        #scaler_range = (self.model.args.data_scale_min, self.model.args.data_scale_max)
        #detuning_scaler = minmax_scaler(scaler_range)
        #control_scaler = minmax_scaler(scaler_range)

        #detuning = detuning - detuning_mean
        #detuning = detuning_scaler.fit_transform(detuning)
        #control = control - control_mean
        #control = control_scaler.fit_transform(control)
        #control = control * mask

        #timeseries = torch.cat([detuning, control, mask], dim=-1)

        timeseries = self.model._fit_dataset.normalize(timeseries)

        # --! having scaled given timeseries, call the scaled version of KIND forward method
        model_output = self._forward_scaled(timeseries)

        # --! extract to-be-unscaled timeseries from the forwarded result
        timeseries_pred = model_output[0]
        timeseries_stat = model_output[1]
        timeseries_trans = model_output[3]

        timeseries_pred = self.model._fit_dataset.denormalize(timeseries_pred)
        timeseries_stat = self.model._fit_dataset.denormalize(timeseries_stat)
        timeseries_trans = self.model._fit_dataset.denormalize(timeseries_trans)

        # --! unscale predicted timeseries
        #
        # --! since the indeces of target dimensions must be present in above dictionaries,
        # --! we use these indeces to directly access the dictionaries

        #timeseries_pred = torch.cat([scaler[k].inverse_transform(timeseries_pred[:, :, [k]]) for k in self.model.args.target_dim], dim=-1)
        #timeseries_pred = torch.cat([timeseries_pred[:, :, [k]] + mean[k] for k in self.model.args.target_dim], dim=-1)

        #timeseries_stat = torch.cat([scaler[k].inverse_transform(timeseries_stat[:, :, [k]]) for k in self.model.args.target_dim], dim=-1)
        #timeseries_stat = torch.cat([timeseries_stat[:, :, [k]] + mean[k] for k in self.model.args.target_dim], dim=-1)

        #timeseries_trans = torch.cat([scaler[k].inverse_transform(timeseries_trans[:, :, [k]]) for k in self.model.args.target_dim], dim=-1)
        #timeseries_trans = torch.cat([timeseries_trans[:, :, [k]] + mean[k] for k in self.model.args.target_dim], dim=-1)

        # --! put unscaled timeseries back to the result tuple and return the tuple
        model_output = list(model_output)
        model_output[0] = timeseries_pred
        model_output[1] = timeseries_stat
        model_output[3] = timeseries_trans

        return tuple(model_output)


class model_fit(model_mode):
    """ Represents a fit mode of a KIND model. """

    def __init__(self, model):
        super().__init__(model)

        # --! create states of this mode
        self._state_stationary_mean          = fit_stationary_mean(self)
        self._state_stationary_uncertainty   = fit_stationary_uncertainty(self)
        self._state_transient_mean           = fit_transient_mean(self)
        self._state_transient_uncertainty    = fit_transient_uncertainty(self)
        self._state                          = self._state_stationary_mean

    def train(self):
        """ Returns immediately as this KIND model is already in fit mode. """
        return

    def eval(self):
        """ Sets this KIND model into evaluation mode. """
        self.model._set_mode(self.model._get_mode_eval())
        self.model._train_module(mode=False)

    def fit_next(self):
        return self.get_state().next()

    def fit(self, dataset):
        """ Fits this KIND model to given ``dataset`` according to the model's current mode and fit state.

        The implementation of this training algorithm resembles the Template pattern, because the method
        defines a structure for training and then calls state classes
        to carry out specific tasks that may vary.
        """

        args = self.model.args

        # --! FIX ME - this may not be the best way to get access to dataset normalization methods
        self.model._fit_dataset = dataset

        # --! make model initializations, data loading, optimizer selection, etc.
        #
        # --! select optimizer after initializing this model!
        self.get_state().init_model()
        train_loader, valid_loader, test_loader = self.get_state().load_data(dataset)
        model_optim = self.select_optimizer()
        early_stopping = utils_nn.early_stopping(patience=args.patience, checkpoint_path=args.checkpoints)

        # --! start training
        for epoch in range(args.nepoch):
            train_loss = []

            for back, fore in train_loader:

                # --! since we compute a full reconstruction loss at the moment, concatenate current
                # --! lookback and forecast windows to get the full timeseries
                # --! to serve as the truth
                truth = torch.cat([back, fore], dim=1)

                # --! extract target dimensions
                truth = truth[:, :, self.model.args.target_dim]

                model_optim.zero_grad()

                # --! forward pass
                loss = self.get_state().compute_loss(truth, self.get_state().forward(back))
                train_loss.append(loss.item())

                # --! backward pass
                loss.backward()

                # --! finalize this training iteration
                model_optim.step()

            train_loss = np.average(train_loss)
            valid_loss = self.validate(valid_loader)
            test_loss = self.validate(test_loader)

            print(f'\tepoch {epoch+1} losses: train={train_loss:.6f}, valid={valid_loss:.6f}, test={test_loss:.6f}')

            # --! use validation loss to check early stopping
            if early_stopping(self.model, valid_loss):
                print("\tearly stopping ...")
                break

            # --! adjust learning rate here

        best_model_path = args.checkpoints + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        return

    def forward(self, timeseries):
        """ Executes a forward pass on scaled ``timeseries`` in fit mode. """
        return self._forward_scaled(timeseries)

    def get_state_stationary_mean(self):
        return self._state_stationary_mean

    def get_state_stationary_uncertainty(self):
        return self._state_stationary_uncertainty

    def get_state_transient_mean(self):
        return self._state_transient_mean

    def get_state_transient_uncertainty(self):
        return self._state_transient_uncertainty

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def validate(self, data_loader):

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
                truth = truth[:, :, self.model.args.target_dim]

                loss = self.get_state().compute_loss(truth, self.get_state().forward(back), validated=True)
                total_loss.append(loss)

        # --! reset this model back to training mode
        self.model.train()

        return np.average(total_loss)

    def select_optimizer(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.model.args.learning_rate,
            weight_decay=self.model.args.weight_decay)


class fit_state(interface):
    """ Manages a state of a KIND model fit mode, e.g. when fitting a stationary mean, or transient uncertainty. """

    def __init__(self, mode):
        super().__init__()

        # --! reference to a fit mode of a KIND model
        self.mode = mode

    @abstractmethod
    def init_model(self):
        """ Initializes this KIND model before training. """
        return

    @abstractmethod
    def load_data(self, dataset):
        """ Loads training, validation and test data from a given ``dataset``. """
        return

    @abstractmethod
    def forward(self, timeseries):
        """ Executes a forward pass of a KIND model. """
        return

    @abstractmethod
    def compute_loss(self, true, pred, validated=False):
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


class fit_stationary_mean(fit_state):

    def __init__(self, mode):
        super().__init__(mode)

    def init_model(self):

        print('>>> train stationary mean >>>')
        model = self.mode.model

        model.transient.freeze_mean()
        model.transient.freeze_var()

        model.stationary.unfreeze()
        model.stationary.freeze_var()

    def load_data(self, dataset):
        return dataset.load(data_type='stat')

    def forward(self, timeseries):
        return self.mode.model(timeseries)

    def compute_loss(self, true, pred, validated=False):

        timeseries = true
        timeseries_pred = pred[1] # < stationary prediction

        fun = pred[5]
        fun_pred = pred[6]

        loss_recon = self.apply_criterion_mean(timeseries, timeseries_pred)
        loss_linear = self.apply_criterion_mean(fun, fun_pred)
        loss = loss_recon + loss_linear

        return loss

    def next(self):
        self.mode.set_state(self.mode.get_state_stationary_uncertainty())
        return True


class fit_stationary_uncertainty(fit_state):

    def __init__(self, mode):
        super().__init__(mode)

    def init_model(self):

        print('>>> train stationary uncertainty >>>')
        model = self.mode.model

        model.transient.freeze_mean()
        model.transient.freeze_var()

        model.stationary.unfreeze()
        model.stationary.freeze_mean()

    def load_data(self, dataset):
        return dataset.load(data_type='mixed')

    def forward(self, timeseries):
        return self.mode.model(timeseries)

    def compute_loss(self, true, pred, validated=False):

        timeseries = true
        timeseries_pred_mean = pred[1]
        timeseries_pred_uncertain = pred[2]
        dfun = pred[10]
        dfun_pred = pred[11]

        # --! when validating, scale model's output
        #
        # --! validation and test data are not initially scaled - the data is scaled internally by the model and
        # --! then scaled back - but for computing uncertainty loss it seems to be more
        # --! straightforward to operate with scaled data
        if validated:
            timeseries = self.mode.model._fit_dataset.normalize(timeseries)
            timeseries_pred_mean = self.mode.model._fit_dataset.normalize(timeseries_pred_mean)
            #mixmax_range = [self.mode.model.args.data_scale_min, self.mode.model.args.data_scale_max]
            #timeseries = utils_data.dataset.scale(utils_data.dataset.demean(timeseries, dim=1), dim=1, minmax=mixmax_range)
            #timeseries_pred_mean = utils_data.dataset.scale(utils_data.dataset.demean(timeseries_pred_mean, dim=1), dim=1, minmax=mixmax_range)

        loss_linear = self.apply_criterion_mean(dfun, dfun_pred)
        loss_uncertain = self.apply_criterion_uncertain(timeseries, timeseries_pred_mean, timeseries_pred_uncertain)
        loss = loss_uncertain + loss_linear

        return loss

    def next(self):
        self.mode.set_state(self.mode.get_state_transient_mean())
        return True


class fit_transient_mean(fit_state):

    def __init__(self, fit):
        super().__init__(fit)

    def init_model(self):

        print('>>> train transient mean >>>')
        model = self.mode.model

        model.transient.unfreeze()
        model.transient.freeze_var()

        model.stationary.freeze_mean()
        model.stationary.freeze_var()

    def load_data(self, dataset):
        return dataset.load(data_type='trans')

    def forward(self, timeseries):
        return self.mode.model(timeseries)

    def compute_loss(self, true, pred, validated=False):

        timeseries = true
        timeseries_pred_mean = pred[3]

        fun = pred[7]
        fun_pred = pred[8]

        loss_recon = self.apply_criterion_mean(timeseries, timeseries_pred_mean)
        loss_linear = self.apply_criterion_mean(fun, fun_pred)
        loss = loss_recon + loss_linear

        return loss

    def next(self):
        self.mode.set_state(self.mode.get_state_transient_uncertainty())
        return True


class fit_transient_uncertainty(fit_state):

    def __init__(self, mode):
        super().__init__(mode)

    def init_model(self):

        print('>>> train transient uncertainty >>>')
        model = self.mode.model

        model.transient.unfreeze()
        model.transient.freeze_mean()

        model.stationary.freeze_mean()
        model.stationary.freeze_var()

    def load_data(self, dataset):
        return dataset.load(data_type='mixed')

    def forward(self, timeseries):
        return self.mode.model(timeseries)

    def compute_loss(self, true, pred, validated=False):

        timeseries = true
        timeseries_pred_mean = pred[3]
        timeseries_pred_uncertain = pred[4]
        dfun = pred[12]
        dfun_pred = pred[13]

        # --! when validating, scale model's output
        #
        # --! validation and test data are not initially scaled - the data is scaled internally by the model and
        # --! then scaled back - but for computing uncertainty loss it seems to be more
        # --! straightforward to operate with scaled data
        if validated:
            timeseries = self.mode.model._fit_dataset.normalize(timeseries)
            timeseries_pred_mean = self.mode.model._fit_dataset.normalize(timeseries_pred_mean)
            #mixmax_range = [self.mode.model.args.data_scale_min, self.mode.model.args.data_scale_max]
            #timeseries = utils_data.dataset.scale(utils_data.dataset.demean(timeseries, dim=1), dim=1, minmax=mixmax_range)
            #timeseries_pred_mean = utils_data.dataset.scale(utils_data.dataset.demean(timeseries_pred_mean, dim=1), dim=1, minmax=mixmax_range)

        loss_linear = self.apply_criterion_mean(dfun_pred, dfun)
        loss_recon = self.apply_criterion_uncertain(timeseries, timeseries_pred_mean, timeseries_pred_uncertain)
        loss = loss_linear + loss_recon

        return loss

    def next(self):
        return False
    

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
        self.nfeature = len(args.feature_dim)
        self.ntarget = len(args.target_dim)
        self.nmask = len(args.mask_dim)
        self.lookback_nsample = args.lookback_nsample
        self.forecast_nsample = args.forecast_nsample

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


class operator_stationary(operator):
    """ Models dynamics of stationary timeseries in a DMD-like manner. """

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
        fun_enc_ni   = self.param_kernsize * (self.nfeature + self.nmask)
        fun_enc_no   = nparam * self.param_kernsize * (self.nfeature + self.nmask)
        fun_enc_feat = utils_nn.make_feat(ni=fun_enc_ni, no=fun_enc_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.fun_enc = utils_nn.fcnn(feat=fun_enc_feat, actfun_hid='relu')

        if (self.nfeature + self.nmask) > 1:
            # --! this linear transformation is supposed to prune the dimensionality of the
            # --! basis functions, such that only the number of these basis functions
            # --! influences the order of the DMD matrix, whereas the number of data
            # --! dimensions has no effect on the order
            fun_prune_ni = nfun * (self.nfeature + self.nmask)
            fun_prune_no = nfun
            self.fun_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)

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
        mod_var_gen_feat = utils_nn.make_feat(ni=mod_var_gen_ni, no=mod_var_gen_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.mod_var_gen = utils_nn.fcnn(feat=mod_var_gen_feat, actfun_hid='relu')

        # --! create prediction decoders to decode predicted embeddings back to timeseries and uncertainty
        pre_dec_ni = nfun
        pre_dec_no = self.param_kernsize * self.ntarget
        pre_dec_feat = utils_nn.make_feat(ni=pre_dec_ni, no=pre_dec_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.pre_mean_dec = utils_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')
        self.pre_var_dec  = utils_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, (self.nfeature + self.nmask))
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * (self.nfeature + self.nmask))

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
            self.param_kernsize * (self.nfeature + self.nmask))
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, (self.nfeature + self.nmask))
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.nfeature + self.nmask)

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
        return self.fun_prune(fun) if (self.nfeature + self.nmask) > 1 else fun

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
        dfun_std     = torch.std(dfun, dim=1, keepdim=True)
        dfun         = (dfun - dfun_mean) / (dfun_std + 1e-8)

        # --! predict the evolution of a function error starting from the first error value upto
        # --! a specified horizon
        dfun_pre, dfun_pre_forecast = self.predict_var(dfun)
        dfun_pre                    = torch.cat([dfun[:, :1], dfun_pre], dim=1)

        # --! denormalize predicted errors to restore original magnitudes, which are essential for
        # --! decoding the right uncertainty magnitudes
        dfun_pre_unsca = torch.cat([dfun_pre, dfun_pre_forecast], dim=1)
        dfun_pre_unsca = dfun_pre_unsca * dfun_std + dfun_mean

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        timeseries_pre_mean  = self.pre_mean_dec(torch.cat([fun_pre, fun_pre_forecast], dim=1))
        timeseries_pre_mean  = timeseries_pre_mean.reshape(timeseries_pre_mean.shape[0], -1, self.ntarget)

        # --! decode predicted and denormalized function errors to model uncertainty
        timeseries_pre_var   = self.pre_var_dec(dfun_pre_unsca)
        timeseries_pre_var   = timeseries_pre_var.reshape(timeseries_pre_var.shape[0], -1, self.ntarget)

        return timeseries_pre_mean, timeseries_pre_var, fun, fun_pre, dfun, dfun_pre

    def freeze_mean(self):
        utils_nn.freeze_module(self.fun_enc)
        if (self.nfeature + self.nmask) > 1:
            utils_nn.freeze_module(self.fun_prune)
        utils_nn.freeze_module(self.mod_mean)
        utils_nn.freeze_module(self.pre_mean_dec)

    def freeze_var(self):
        utils_nn.freeze_module(self.mod_var_gen)
        utils_nn.freeze_module(self.pre_var_dec)

    def unfreeze(self):
        utils_nn.unfreeze_module(self.fun_enc)
        if (self.nfeature + self.nmask) > 1:
            utils_nn.unfreeze_module(self.fun_prune)
        utils_nn.unfreeze_module(self.mod_mean)
        utils_nn.unfreeze_module(self.mod_var_gen)
        utils_nn.unfreeze_module(self.pre_mean_dec)
        utils_nn.unfreeze_module(self.pre_var_dec)


class operator_transient(operator):
    """Models dynamics of transient timeseries using a Transformer-based attention mechanism."""

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
        fun_enc_ni   = self.param_kernsize * (self.nfeature + self.nmask)
        fun_enc_no   = nparam * self.param_kernsize * (self.nfeature + self.nmask)
        fun_enc_feat = utils_nn.make_feat(ni=fun_enc_ni, no=fun_enc_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.fun_enc = utils_nn.fcnn(feat=fun_enc_feat, actfun_hid='relu')

        if (self.nfeature + self.nmask) > 1:
            # --! this linear transformation is supposed to prune the dimensionality of the
            # --! basis functions, such that only the number of these basis functions
            # --! influences the order of linear matrices and the number of data
            # --! dimensions has no effect on that order
            fun_prune_ni = nfun * (self.nfeature + self.nmask)
            fun_prune_no = nfun
            self.fun_prune = torch.nn.Linear(fun_prune_ni, fun_prune_no, bias=False)

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
        mod_mean_gen_feat = utils_nn.make_feat(ni=mod_mean_gen_ni, no=mod_mean_gen_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.mod_mean_gen = utils_nn.fcnn(feat=mod_mean_gen_feat, actfun_hid='relu')

        # --! create a generator that produces models capturing the evolution of variance (uncertainty)
        #
        # --! the generator takes a flattened sequence of function values and returns
        # --! a flattened sequence of square matrices
        mod_var_gen_ni = fun_nsample * nfun
        mod_var_gen_no = (fun_nsample + self.fun_nsample_forecast) * nfun * nfun
        mod_var_gen_feat = utils_nn.make_feat(ni=mod_var_gen_ni, no=mod_var_gen_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.mod_var_gen = utils_nn.fcnn(feat=mod_var_gen_feat, actfun_hid='relu')

        # --! create MLP-based decoders to decode embeddings back to timeseries with uncertainty
        pre_dec_ni = nfun
        pre_dec_no = self.param_kernsize * self.ntarget
        pre_dec_feat = utils_nn.make_feat(ni=pre_dec_ni, no=pre_dec_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.pre_mean_dec = utils_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')
        self.pre_var_dec = utils_nn.fcnn(feat=pre_dec_feat, actfun_hid='relu')

    def embed(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.param_kernsize, self.nfeature + self.nmask)
        i = i.reshape(timeseries.shape[0], -1, self.param_kernsize * (self.nfeature + self.nmask))

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
            self.param_kernsize * (self.nfeature + self.nmask))
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.param_kernsize, self.nfeature + self.nmask)
        i = i.reshape(i.shape[0], i.shape[1], self.param_kernsize, self.nfeature + self.nmask)

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
        return self.fun_prune(fun) if (self.nfeature + self.nmask) > 1 else fun

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
        dfun_std     = torch.std(dfun, dim=1, keepdim=True)
        dfun         = (dfun - dfun_mean) / (dfun_std + 1e-8)
        #dfun         = dfun - dfun_mean
        #scaler       = minmax_scaler(feature_range=(-1, 1))
        #dfun         = scaler.fit_transform(dfun)

        # --! predict the evolution of a function error starting from the first error value upto
        # --! a specified horizon
        dfun_pre, dfun_pre_forecast = self.predict_var(dfun)
        dfun_pre                    = torch.cat([dfun[:, :1], dfun_pre], dim=1)

        # --! denormalize predicted errors to restore original magnitudes, which are essential for
        # --! decoding the right uncertainty magnitudes
        dfun_pre_unsca = torch.cat([dfun_pre, dfun_pre_forecast], dim=1)
        dfun_pre_unsca = dfun_pre_unsca * dfun_std + dfun_mean
        #dfun_pre_unsca = scaler.inverse_transform(torch.cat([dfun_pre, dfun_pre_forecast], dim=1))
        #dfun_pre_unsca = dfun_pre_unsca + dfun_mean

        # --! decode predicted embeddings to timeseries
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        timeseries_pre_mean = self.pre_mean_dec(torch.cat([fun_pre, fun_pre_forecast], dim=1))
        timeseries_pre_mean = timeseries_pre_mean.reshape(timeseries_pre_mean.shape[0], -1, self.ntarget)

        # --! decode predicted and denormalized function errors to model uncertainty (variance)
        timeseries_pre_var  = self.pre_var_dec(dfun_pre_unsca)
        timeseries_pre_var  = timeseries_pre_var.reshape(timeseries_pre_var.shape[0], -1, self.ntarget)

        return timeseries_pre_mean, timeseries_pre_var, fun, fun_pre, dfun, dfun_pre

    def freeze_mean(self):
        utils_nn.freeze_module(self.fun_enc)
        if (self.nfeature + self.nmask) > 1:
            utils_nn.freeze_module(self.fun_prune)
        utils_nn.freeze_module(self.mod_mean_att_enc)
        utils_nn.freeze_module(self.mod_mean_gen)
        utils_nn.freeze_module(self.pre_mean_dec)

    def freeze_var(self):
        utils_nn.freeze_module(self.mod_var_gen)
        utils_nn.freeze_module(self.pre_var_dec)

    def unfreeze(self):
        utils_nn.unfreeze_module(self.fun_enc)
        if (self.nfeature + self.nmask) > 1:
            utils_nn.unfreeze_module(self.fun_prune)
        utils_nn.unfreeze_module(self.mod_mean_att_enc)
        utils_nn.unfreeze_module(self.mod_mean_gen)
        utils_nn.unfreeze_module(self.mod_var_gen)
        utils_nn.unfreeze_module(self.pre_mean_dec)
        utils_nn.unfreeze_module(self.pre_var_dec)

