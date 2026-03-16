# --! Implementation of Kalman-inspired neural decomposition, or KIND --!

from abc import abstractmethod
from abc import ABC as interface

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import json

from collections import namedtuple

import util_data
import util_nn


regimes = namedtuple('regimes', 'nominal excursion')
model_output = namedtuple('model_output', [
    'blend', 'alpha',
    'mean_nom', 'zeta_raw_nom', 'zeta_nom',
    'mean_exc', 'zeta_raw_exc', 'zeta_exc',
    'mean_embed_nom', 'mean_embed_pred_nom', 'zeta_embed_nom', 'zeta_embed_pred_nom',
    'mean_embed_exc', 'mean_embed_pred_exc', 'zeta_embed_exc', 'zeta_embed_pred_exc'])


def create_args_parser():
    """ Creates a command-line parser for KIND arguments. """
    parser = argparse.ArgumentParser(description='KIND: learned hybrid dynamics')

    # --! data arguments
    parser.add_argument('--file_dir', type=str, required=True, help='relative path to file directory')
    parser.add_argument('--file_name', type=str, required=True, help='file name with no extension (e.g., .csv) and no suffix (e.g., _stat)')
    parser.add_argument('--file_index', type=int, required=False, default=0, help='file index to define or separate learning stages')
    parser.add_argument('--file_ext', type=str, required=False, default='.csv', help='data file extension')
    parser.add_argument('--data_nsample_all', type=int, required=True, help='number of samples in time series stored in all data')
    parser.add_argument('--data_nsample_nom', type=int, required=True, help='number of samples in time series stored in nominal data')
    parser.add_argument('--data_nsample_exc', type=int, required=True, help='number of samples in time series stored in excursion data')
    parser.add_argument('--data_train_size', type=float, required=False, default=0.8, help='dataset part to include in training')
    parser.add_argument('--data_test_size', type=float, required=False, default=0.5, help='non-train part to include in test, rest is validation')
    parser.add_argument('--feature_ndim', type=int, required=True, help='number of feature dimensions in data')
    parser.add_argument('--target_ndim', type=int, required=True, help='number of target dimensions in data')

    # --! KIND
    parser.add_argument('--back_nsample', type=int, required=True, help='number of samples in lookback window')
    parser.add_argument('--fore_nsample', type=int, required=True, help='number of samples in forecast window')
    parser.add_argument('--rez_nsample_nom', type=int, required=True, help='resolution (number of samples) in nominal data window')
    parser.add_argument('--rez_nsample_exc', type=int, required=True, help='resolution (number of samples) in excursion data window')
    parser.add_argument('--embed_nom', type=json.loads, required=True, help='type and number of embeddings for nominal operator')
    parser.add_argument('--embed_exc', type=json.loads, required=True, help='type and number of embeddings for excursion operator')

    # --! training
    parser.add_argument('--batch_size', type=int, required=False, default=128, help='training batch size')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001, help='learning rate during training')
    parser.add_argument('--weight_decay', type=float, required=False, default=0.0001, help='weight decay to regularize training')
    parser.add_argument('--nepoch', type=int, required=False, default=100, help='number of training epochs')
    parser.add_argument('--patience', type=int, required=False, default=10, help='patience for early stopping during training')
    parser.add_argument('--checkpoints', type=str, required=True, help='location of model training checkpoints')

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

        # --! execute both operators on given lookback
        mean_nom, raw_zeta_nom, mean_embed_nom, mean_embed_pred_nom, zeta_embed_nom, zeta_embed_pred_nom = self.model_nom(lookback)
        mean_exc, raw_zeta_exc, mean_embed_exc, mean_embed_pred_exc, zeta_embed_exc, zeta_embed_pred_exc = self.model_exc(lookback)

        # --! raw zeta signal has two features: signed error and geometry error
        zeta_chan_ndim = self.args.target_ndim
        signed_error_nom, geometry_error_nom = torch.split(raw_zeta_nom, [zeta_chan_ndim, zeta_chan_ndim], dim=-1)
        signed_error_exc, geometry_error_exc = torch.split(raw_zeta_exc, [zeta_chan_ndim, zeta_chan_ndim], dim=-1)

        # --! convert raw zeta to true zeta - a non-negative signal with penalty on high-angle intersections
        beta = 2.0 # <-- todo: put beta into model arguments
        zeta_nom = torch.abs(signed_error_nom) + beta * torch.abs(geometry_error_nom)
        zeta_exc = torch.abs(signed_error_exc) + beta * torch.abs(geometry_error_exc)

        # --! derive alpha [0, 1]
        alpha = zeta_exc / (zeta_exc + zeta_nom)

        # --! blend the two types of time series using the derived alpha to get the final prediction
        prediction = alpha * mean_nom + (1 - alpha) * mean_exc

        return model_output(

            blend=prediction, alpha=alpha,

            mean_nom=mean_nom, zeta_raw_nom=raw_zeta_nom, zeta_nom=zeta_nom,
            mean_exc=mean_exc, zeta_raw_exc=raw_zeta_exc, zeta_exc=zeta_exc,

            mean_embed_nom=mean_embed_nom, mean_embed_pred_nom=mean_embed_pred_nom,
            mean_embed_exc=mean_embed_exc, mean_embed_pred_exc=mean_embed_pred_exc,

            zeta_embed_nom=zeta_embed_nom, zeta_embed_pred_nom=zeta_embed_pred_nom,
            zeta_embed_exc=zeta_embed_exc, zeta_embed_pred_exc=zeta_embed_pred_exc)


class model_adapter:
    """Adapts a KIND model to data normalization."""

    def __init__(self, model, normalizer):

        # --! freeze model
        model.eval()
        util_nn.freeze_module(model)

        self.model = model
        self.normalizer = normalizer

    def __call__(self, lookback):
        return self.forward(lookback)

    def forward(self, lookback):

        # --! normalize input data
        lookback = self.normalizer.normalize(lookback)

        # --! pass normalized data to the model
        model_o = self.model(lookback)

        # --! denormalize model output
        #
        # --! extract predictions that need to be denormalized
        pred = model_o.blend
        pred_nom = model_o.mean_nom
        pred_exc = model_o.mean_exc

        # --! denormalize extracted predictions
        pred = self.normalizer.denormalize(pred)
        pred_nom = self.normalizer.denormalize(pred_nom)
        pred_exc = self.normalizer.denormalize(pred_exc)

        # --! put unscaled timeseries back to the result tuple and return the tuple
        model_o = model_o._asdict()
        model_o['blend'] = pred
        model_o['mean_nom'] = pred_nom
        model_o['mean_exc'] = pred_exc

        return model_output(**model_o)

    @property
    def args(self):
        return self.model.args


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

        # --! with gradients stopped, run validation loop
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

    def apply_criterion_mean(self, true, pred):
        criterion = torch.nn.MSELoss(reduction='mean')
        return criterion(pred, true)

    def apply_criterion_zeta(self, true, pred, zeta):

        zeta_chan_ndim = self.training.model.args.target_ndim
        signed_error, geometry_error = torch.split(zeta, [zeta_chan_ndim, zeta_chan_ndim], dim=-1)

        # --! take difference between true and predicted time series,
        # --! hence zero difference at a certain time step signifies that the time series intersect
        with torch.no_grad():
            signed_target = true - pred

        signed_loss = self.apply_criterion_mean(signed_target, signed_error)

        # --! take difference between derivatives of true and predicted time series,
        # --! hence zero difference at a certain time step signifies that the time series change in a similar way
        with torch.no_grad():
            da = self.smooth_derivative(true, window_nsample=9) # <-- fix me: put number of window samples into model arguments
            db = self.smooth_derivative(pred, window_nsample=9)
            geometry_target = da - db

        geometry_loss = self.apply_criterion_mean(geometry_target, geometry_error)

        return signed_loss + geometry_loss

    def smooth_derivative(self, x, window_nsample=9):

        # --! compute finite difference along time dimension
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx = F.pad(dx, (0, 0, 1, 0)) # pad time dimension

        nbatch, nsample, nchan = dx.shape

        # --! create a moving-average kernel
        kernel = torch.ones(window_nsample, device=x.device, dtype=x.dtype) / window_nsample
        kernel = kernel.view(1, 1, window_nsample)

        # --! prepare for grouped convolution: [nbatch, nchan, nsample]
        dx = dx.transpose(1, 2)

        # Repeat kernel for each channel
        kernel = kernel.repeat(nchan, 1, 1)

        # --! perform depthwise convolution
        dx = F.conv1d(
            dx,
            kernel,
            padding=window_nsample // 2,
            groups=nchan
        )

        # --! back to [nbatch, nsample, nchan]
        dx = dx.transpose(1, 2)

        return dx


class mean_training_nom(training_phase):

    def __init__(self, training):
        super().__init__(training)

    def init_model(self):

        print('>>> training nominal mean >>>')

        model = self.training.model

        model.model_exc.freeze_mean()
        model.model_exc.freeze_zeta()

        model.model_nom.unfreeze()
        model.model_nom.freeze_zeta()

    def load_data(self, dataset):
        return dataset.load(data_type='nom')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_mean = pred.mean_nom

        embed = pred.mean_embed_nom
        embed_pred = pred.mean_embed_pred_nom

        loss_recon = self.apply_criterion_mean(timeseries, timeseries_mean)
        loss_linear = self.apply_criterion_mean(embed, embed_pred)
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
        model.model_exc.freeze_zeta()

        model.model_nom.unfreeze()
        model.model_nom.freeze_mean()

    def load_data(self, dataset):
        return dataset.load(data_type='mixed')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_mean = pred.mean_nom
        timeseries_zeta = pred.zeta_raw_nom

        embed = pred.zeta_embed_nom
        embed_pred = pred.zeta_embed_pred_nom

        loss_linear = self.apply_criterion_mean(embed, embed_pred)
        loss_zeta = self.apply_criterion_zeta(timeseries, timeseries_mean, timeseries_zeta)
        loss = loss_zeta + loss_linear

        return loss

    def next(self):
        self.training.set_phase(self.training.get_phase_mean_exc())
        return True


class mean_training_exc(training_phase):

    def __init__(self, training):
        super().__init__(training)

    def init_model(self):

        print('>>> training excursion mean >>>')

        model = self.training.model

        model.model_exc.unfreeze()
        model.model_exc.freeze_zeta()

        model.model_nom.freeze_mean()
        model.model_nom.freeze_zeta()

    def load_data(self, dataset):
        return dataset.load(data_type='exc')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_mean = pred.mean_exc

        embed = pred.mean_embed_exc
        embed_pred = pred.mean_embed_pred_exc

        loss_recon = self.apply_criterion_mean(timeseries, timeseries_mean)
        loss_linear = self.apply_criterion_mean(embed, embed_pred)
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
        model.model_nom.freeze_zeta()

    def load_data(self, dataset):
        return dataset.load(data_type='mixed')

    def forward(self, lookback):
        return self.training.model(lookback)

    def compute_loss(self, true, pred):

        timeseries = true
        timeseries_mean = pred.mean_exc
        timeseries_zeta = pred.zeta_raw_exc

        embed = pred.zeta_embed_exc
        embed_pred = pred.zeta_embed_pred_exc

        loss_linear = self.apply_criterion_mean(embed, embed_pred)
        loss_uncertain = self.apply_criterion_zeta(timeseries, timeseries_mean, timeseries_zeta)
        loss = loss_uncertain + loss_linear

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
        self.nfeature = args.feature_ndim
        self.ntarget = args.target_ndim
        self.back_nsample = args.back_nsample
        self.fore_nsample = args.fore_nsample

    @abstractmethod
    def embed_mean(self, timeseries):
        """Embeds ``timeseries`` in a mean latent space to model the mean of these ``timeseries``."""
        return

    @abstractmethod
    def predict_mean(self, embeddings):
        """Predicts ``embeddings`` in the mean latent space."""
        return

    @abstractmethod
    def embed_zeta(self, timeseries):
        """Embeds ``timeseries`` in a zeta latent space to model how uncertain the model is about these ``timeseries``."""
        return

    @abstractmethod
    def predict_zeta(self, embeddings):
        """Predicts ``embeddings`` in the zeta latent space."""
        return

    @abstractmethod
    def forward(self, timeseries):
        """Predicts given ``timeseries`` based on the mean and zeta latent spaces."""
        return

    @abstractmethod
    def freeze_mean(self):
        """Makes submodules that are responsible for mean prediction fixed, untrainable."""
        return

    @abstractmethod
    def freeze_zeta(self):
        """Makes submodules that are responsible for the zeta prediction fixed, untrainable."""
        return

    @abstractmethod
    def unfreeze(self):
        """Makes all trainable submodules trainable again."""
        return

    def _extract_embed_config(self, config):
        """Extracts embedding configuration from ``config`` to a format that facilitates model internal workings.

        The extraction converts the number of specific embeddings, e.g., 'sin', into
        the total number of parameters required by these specific embeddings. For
        example, if 5 embeddings of 'sin' are to be used, then the total number
        of parameters becomes 5 * 2 = 10, as a 'sin' takes 2 parameters.
        """

        newconfig = config.copy()

        for embed in newconfig:
            if embed == 'sin':
                newconfig[embed] = newconfig[embed] * self.sin_nparam # sine takes two parameters
            elif embed == 'cos':
                newconfig[embed] = newconfig[embed] * self.cos_nparam # cosine takes two parameters

        return newconfig

    def _eval_embed(self, embed, param):
        if embed == 'sin':
            return self._eval_sin(param)
        elif embed == 'cos':
            return self._eval_cos(param)
        elif embed == 'data':
            return self._eval_data(param)
        elif embed == 'poly':
            return self._eval_poly(param)
        elif embed == 'exp':
            return self._eval_exp(param)
        else:
            raise Exception("unsupported embedding!")

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
    """Models dynamics of nominal time series in a dynamical mode decomposition-like manner."""

    def __init__(self, args):

        # --! initialize common operator parameters
        super().__init__(args)

        # --! get the number of embeddings
        nembed = sum(args.embed_nom.values())

        # --! initialize nominal-specific parameters
        self.embed_config = self._extract_embed_config(args.embed_nom)
        self.rez_nsample = args.rez_nsample_nom

        if self.fore_nsample % self.rez_nsample:
            raise Exception('number of forecast samples must be a multiple of resolution!')

        # --! get the number of embedded samples in the forecast window
        self.embed_nsample_fore = self.fore_nsample // self.rez_nsample

        # --! get the total number of required embedded parameters
        nparam = sum(self.embed_config.values())

        # --! get the number of embedded samples in the lookback window
        embed_nsample_back = self.back_nsample // self.rez_nsample

        # --! create an encoder to encode timeseries into embedded values
        #
        # --! more precisely, input timeseries are partitioned into segments according to the resolution, and
        # --! the encoder encodes segment-specific kernels for every embedded parameter
        # --! in the embedded (latent) space
        enc_ni = self.rez_nsample * self.nfeature
        enc_no = nparam * self.rez_nsample * self.nfeature
        enc_feat = util_nn.make_feat(ni=enc_ni, no=enc_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.enc_mean = util_nn.fcnn(feat=enc_feat, actfun_hid='relu')
        self.enc_zeta = util_nn.fcnn(feat=enc_feat, actfun_hid='relu')

        if self.nfeature > 1:
            # --! this linear transformation is supposed to prune the dimensionality of the embeddings,
            # --! such that only the number of these embeddings influences
            # --! the order of the DMD matrix, whereas the number of
            # --! data dimensions has no effect on the order
            prune_ni = nembed * self.nfeature
            prune_no = nembed
            self.prune_mean = torch.nn.Linear(prune_ni, prune_no, bias=False)
            self.prune_zeta = torch.nn.Linear(prune_ni, prune_no, bias=False)

        # --! create a DMD-like model (a matrix) that captures nominal (mean) dynamics
        #
        # --! since this matrix is learned only once and does not adapt during runtime,
        # --! this operator can also be called static, instead of nominal
        model_ni = nembed
        model_no = nembed
        self.model_mean = torch.nn.Linear(model_ni, model_no, bias=False)

        # --! create a generator that produces models capturing the evolution of uncertainty zeta
        #
        # --! the generator takes a flattened sequence of embedded values and returns
        # --! a flattened sequence of square matrices
        model_ni = embed_nsample_back * nembed
        model_no = (embed_nsample_back + self.embed_nsample_fore) * nembed * nembed
        model_feat = util_nn.make_feat(ni=model_ni, no=model_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.model_zeta = util_nn.fcnn(feat=model_feat, actfun_hid='relu')

        # --! create prediction decoder to decode predicted embeddings back to timeseries
        dec_ni = nembed
        dec_no = self.rez_nsample * self.ntarget
        dec_feat = util_nn.make_feat(ni=dec_ni, no=dec_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.dec_mean = util_nn.fcnn(feat=dec_feat, actfun_hid='relu')

        # --! construct uncertainty decoder that outputs
        # --! two uncertainty features for each target: signed error and error 'geometry'
        dec_ni = nembed
        dec_no = self.rez_nsample * self.ntarget * 2
        dec_feat = util_nn.make_feat(ni=dec_ni, no=dec_no, nneuron=args.nneuron_stat, nlayer=args.nlayer_stat)
        self.dec_zeta = util_nn.fcnn(feat=dec_feat, actfun_hid='relu')

    def embed_mean(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.rez_nsample, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.rez_nsample * self.nfeature)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.enc_mean(i)

        nparam = sum(self.embed_config.values())

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
            self.rez_nsample * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.rez_nsample, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.rez_nsample, self.nfeature)

        # --! with the help of kernels extract function parameters from input timeseries
        param = torch.einsum("blkdf, bldf -> blfk", kern, i)

        # --! split extracted parameters based on the number of parameters required by each
        # --! type of grouped basis functions
        param = torch.split(param, list(self.embed_config.values()), dim=-1)

        # --! evaluate embeddings at each slice of timeseries
        embed = torch.cat([self._eval_embed(e, p) for e, p in zip(self.embed_config.keys(), param)], dim=-1)

        # --! reshape dimensions to go from a shape [B, T / kernsize, ndim, nembed]
        # --! to [B, T / kernsize, ndim * nembed]
        embed = embed.reshape(embed.shape[0], embed.shape[1], -1)

        # --! prune extra dimensionality caused by multidimensional data
        return self.prune_mean(embed) if self.nfeature > 1 else embed

    def embed_zeta(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.rez_nsample, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.rez_nsample * self.nfeature)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.enc_zeta(i)

        nparam = sum(self.embed_config.values())

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
            self.rez_nsample * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.rez_nsample, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.rez_nsample, self.nfeature)

        # --! with the help of kernels extract function parameters from input timeseries
        param = torch.einsum("blkdf, bldf -> blfk", kern, i)

        # --! split extracted parameters based on the number of parameters required by each
        # --! type of grouped basis functions
        param = torch.split(param, list(self.embed_config.values()), dim=-1)

        # --! evaluate embeddings at each slice of timeseries
        embed = torch.cat([self._eval_embed(e, p) for e, p in zip(self.embed_config.keys(), param)], dim=-1)

        # --! reshape dimensions to go from a shape [B, T / kernsize, ndim, nembed]
        # --! to [B, T / kernsize, ndim * nembed]
        embed = embed.reshape(embed.shape[0], embed.shape[1], -1)

        # --! prune extra dimensionality caused by multidimensional data
        return self.prune_zeta(embed) if self.nfeature > 1 else embed

    def predict_mean(self, embeddings):

        back_nsample = embeddings.shape[1]
        fore_nsample = self.embed_nsample_fore

        horizon = back_nsample + fore_nsample

        # --! extract the matrix of stationary dynamics
        #
        # --! the matrix is unsqueezed to a shape [1, nfun, nfun] to allow
        # --! broadcasting when multiplying with functions
        mat = torch.unsqueeze(self.model_mean.weight, 0)

        # --! stack together matrices raised to powers to cover all prediction horizon
        #
        # --! powered matrices are shaped as [B, H - 1, nfun, nfun], where H is the number of horizon steps
        # --! and -1 is because we do not predict the first time step
        #
        # --! note that we omit matrix transpose here, relying on training to figure it out
        mat_power = torch.stack([
            torch.linalg.matrix_power(mat, power) for power in range(1, horizon)], dim=1)

        # --! extract the initial conditions of embedded values, i.e. the first slice
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        embed_ic = torch.unsqueeze(embeddings[:, :1], -2)

        # --! predict the stationary evolution of embedded values by multiplying their initial conditions
        # --! by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun], i.e.
        # --! batches with functions trajectories consisting of individual function-points [1, nfun]
        embed_pred = torch.matmul(embed_ic, mat_power)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        embed_pred = torch.squeeze(embed_pred, -2)

        return torch.split(embed_pred, [back_nsample - 1, fore_nsample], dim=1)

    def predict_zeta(self, embeddings):

        # --! constants for convenience
        batsize = embeddings.shape[0]
        embed_nsample = embeddings.shape[1]
        nembed = embeddings.shape[2]

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of embedded values
        model_i = torch.flatten(embeddings, start_dim=1)
        mat = self.model_zeta(model_i).reshape(batsize, -1, nembed, nembed)
        mat = util_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial condition of embedded history
        #
        # --! the initial condition (ic) is shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by the matrices
        embed_ic = torch.unsqueeze(embeddings[:, :1], -2)

        # --! predict the evolution of embeddings by multiplying their initial conditions by the matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nfun],
        # --! i.e. a batch with embedded trajectories consisting of individual zeta-points [1, nembed]
        embed_pred = torch.matmul(embed_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        embed_pred = torch.squeeze(embed_pred, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(embed_pred, [embed_nsample - 1, self.embed_nsample_fore], dim=1)

    def forward(self, timeseries):

        # --! from given timeseries, embed mean latent values and then predict these embeddings
        # --! starting from the first value (initial condition) upto a specified horizon
        mean = self.embed_mean(timeseries)
        mean_pred_back, mean_pred_fore  = self.predict_mean(mean)
        mean_pred_back = torch.cat([mean[:, :1], mean_pred_back], dim=1)

        # --! do a similar procedure for uncertainty score zeta
        zeta = self.embed_zeta(timeseries)
        zeta_pred_back, zeta_pred_fore = self.predict_zeta(zeta)
        zeta_pred_back = torch.cat([zeta[:, :1], zeta_pred_back], dim=1)

        # --! decode predicted mean embeddings to time series domain
        #
        # --! timeseries are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        ts_mean = self.dec_mean(torch.cat([mean_pred_back, mean_pred_fore], dim=1))
        ts_mean = ts_mean.reshape(ts_mean.shape[0], -1, self.ntarget)

        # --! decode predicted zeta embeddings to time series domain
        ts_zeta = self.dec_zeta(torch.cat([zeta_pred_back, zeta_pred_fore], dim=1))
        ts_zeta = ts_zeta.reshape(ts_zeta.shape[0], -1, self.ntarget * 2)

        return ts_mean, ts_zeta, mean, mean_pred_back, zeta, zeta_pred_back

    def freeze_mean(self):
        util_nn.freeze_module(self.enc_mean)
        if self.nfeature > 1:
            util_nn.freeze_module(self.prune_mean)
        util_nn.freeze_module(self.model_mean)
        util_nn.freeze_module(self.dec_mean)

    def freeze_zeta(self):
        util_nn.freeze_module(self.enc_zeta)
        if self.nfeature > 1:
            util_nn.freeze_module(self.prune_zeta)
        util_nn.freeze_module(self.model_zeta)
        util_nn.freeze_module(self.dec_zeta)

    def unfreeze(self):
        util_nn.unfreeze_module(self.enc_mean)
        util_nn.unfreeze_module(self.enc_zeta)
        if self.nfeature > 1:
            util_nn.unfreeze_module(self.prune_mean)
            util_nn.unfreeze_module(self.prune_zeta)
        util_nn.unfreeze_module(self.model_mean)
        util_nn.unfreeze_module(self.model_zeta)
        util_nn.unfreeze_module(self.dec_mean)
        util_nn.unfreeze_module(self.dec_zeta)


class operator_exc(operator):
    """Models dynamics of excursion timeseries using a Transformer-based attention mechanism."""

    def __init__(self, args):

        # --! initialize common operator parameters
        super().__init__(args)

        # --! get the total number of embeddings
        nembed = sum(args.embed_exc.values())

        # --! initialize excursion-specific parameters
        self.embed_config = self._extract_embed_config(args.embed_exc)
        self.rez_nsample = args.rez_nsample_exc

        if self.fore_nsample % self.rez_nsample:
            raise Exception('number of forecast samples must be a multiple of resolution!')
        self.embed_nsample_fore  = self.fore_nsample // self.rez_nsample

        # --! get the total number of required embedded parameters
        nparam = sum(self.embed_config.values())

        # --! get the number of embedded samples in the lookback window
        embed_nsample_back = self.back_nsample // self.rez_nsample

        # --! create an MLP-based encoder to encode timeseries into embeddings
        #
        # --! more precisely, input timeseries are partitioned into segments according to resolution, and the encoder
        # --! encodes segment-specific kernels for every embedded parameter
        # --! in the embedded (latent) space
        enc_ni = self.rez_nsample * self.nfeature
        enc_no = nparam * self.rez_nsample * self.nfeature
        enc_feat = util_nn.make_feat(ni=enc_ni, no=enc_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.enc_mean = util_nn.fcnn(feat=enc_feat, actfun_hid='relu')
        self.enc_zeta = util_nn.fcnn(feat=enc_feat, actfun_hid='relu')

        if self.nfeature > 1:
            # --! this linear transformation is supposed to prune the dimensionality of the embeddings,
            # --! such that only the number of these embeddings influences
            # --! the order of linear matrices and the number of data
            # --! dimensions has no effect on that order
            prune_ni = nembed * self.nfeature
            prune_no = nembed
            self.prune_mean = torch.nn.Linear(prune_ni, prune_no, bias=False)
            self.prune_zeta = torch.nn.Linear(prune_ni, prune_no, bias=False)

        # --! encoder network which learns to attend over embedded values, where the encoder
        # --! is implemented in terms of a Transformer encoder network
        enc_ni = nembed
        self.enc_model_mean = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=enc_ni,
                nhead=1,
                dim_feedforward=args.nneuron_trans,
                batch_first=True),
            num_layers=args.nlayer_trans,
            enable_nested_tensor=False)

        # --! an additional generator to create linear time-varying matrices from
        # --! encoded attention; these matrices enable piecewise-linear
        # --! predictions of mean embedding sequences
        #
        # --! the generator takes a flattened sequence of embedded values and
        # --! returns a flattened sequence of square matrices,
        # --! covering all prediction horizon
        model_ni = embed_nsample_back * nembed
        model_no = (embed_nsample_back + self.embed_nsample_fore) * nembed * nembed
        model_feat = util_nn.make_feat(ni=model_ni, no=model_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.model_mean = util_nn.fcnn(feat=model_feat, actfun_hid='relu')

        # --! create a generator that produces models capturing the evolution of uncertainty zeta
        #
        # --! the generator takes a flattened sequence of embedded values and returns
        # --! a flattened sequence of square matrices
        model_ni = embed_nsample_back * nembed
        model_no = (embed_nsample_back + self.embed_nsample_fore) * nembed * nembed
        model_feat = util_nn.make_feat(ni=model_ni, no=model_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.model_zeta = util_nn.fcnn(feat=model_feat, actfun_hid='relu')

        # --! create MLP-based decoders to decode embeddings back to timeseries
        dec_ni = nembed
        dec_no = self.rez_nsample * self.ntarget
        dec_feat = util_nn.make_feat(ni=dec_ni, no=dec_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.dec_mean = util_nn.fcnn(feat=dec_feat, actfun_hid='relu')

        # --! construct uncertainty decoder that outputs
        # --! two uncertainty features for each target: signed error and error 'geometry'
        dec_ni = nembed
        dec_no = self.rez_nsample * self.ntarget * 2
        dec_feat = util_nn.make_feat(ni=dec_ni, no=dec_no, nneuron=args.nneuron_trans, nlayer=args.nlayer_trans)
        self.dec_zeta = util_nn.fcnn(feat=dec_feat, actfun_hid='relu')

    def embed_mean(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.rez_nsample, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.rez_nsample * self.nfeature)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.enc_mean(i)

        nparam = sum(self.embed_config.values())

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
            self.rez_nsample * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.rez_nsample, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.rez_nsample, self.nfeature)

        # --! with the help of kernels extract function parameters from input timeseries
        param = torch.einsum("blkdf, bldf -> blfk", kern, i)

        # --! split extracted parameters based on number of parameters required by each basis function
        param = torch.split(param, list(self.embed_config.values()), dim=-1)

        # --! evaluate embeddings at each slice of timeseries
        #
        # --! note that there is one single measurement of each embedding to describe
        # --! a slice, so the granularity of slicing plays an important a role
        embed = torch.cat([self._eval_embed(e, p) for e, p in zip(self.embed_config.keys(), param)], dim=-1)

        # --! reshape dimensions to go from a shape [B, T / kernsize, ndim, nembed]
        # --! to [B, T / kernsize, ndim * nembed]
        embed = embed.reshape(embed.shape[0], embed.shape[1], -1)

        # --! prune extra dimensionality caused by multidimensional data
        return self.prune_mean(embed) if self.nfeature > 1 else embed

    def embed_zeta(self, timeseries):

        # --! reshape timeseries to form an input to embeddings encoder
        #
        # --! the length of timeseries is reshaped to put kernels into the prelast dimension, i.e.
        # --! [B, T, ndim] -> [B, T / kernsize, kernsize, ndim], where
        # --! B, T and ndim are the number of batch elements,
        # --! timesteps and data channels, respectively
        #
        # --! note that -1 below infers the size of a dimension
        i = timeseries.reshape(timeseries.shape[0], -1, self.rez_nsample, self.nfeature)
        i = i.reshape(timeseries.shape[0], -1, self.rez_nsample * self.nfeature)

        # --! based on inputs, encode parameter kernels (filters)
        kern = self.enc_zeta(i)

        nparam = sum(self.embed_config.values())

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
            self.rez_nsample * self.nfeature)
        kern = kern.reshape(
            i.shape[0], i.shape[1],
            nparam,
            self.rez_nsample, self.nfeature)
        i = i.reshape(i.shape[0], i.shape[1], self.rez_nsample, self.nfeature)

        # --! with the help of kernels extract function parameters from input timeseries
        param = torch.einsum("blkdf, bldf -> blfk", kern, i)

        # --! split extracted parameters based on the number of parameters required by each
        # --! type of grouped basis functions
        param = torch.split(param, list(self.embed_config.values()), dim=-1)

        # --! evaluate embeddings at each slice of timeseries
        embed = torch.cat([self._eval_embed(e, p) for e, p in zip(self.embed_config.keys(), param)], dim=-1)

        # --! reshape dimensions to go from a shape [B, T / kernsize, ndim, nembed]
        # --! to [B, T / kernsize, ndim * nembed]
        embed = embed.reshape(embed.shape[0], embed.shape[1], -1)

        # --! prune extra dimensionality caused by multidimensional data
        return self.prune_zeta(embed) if self.nfeature > 1 else embed

    def predict_mean(self, embeddings):

        batsize = embeddings.shape[0]
        embed_nsample = embeddings.shape[1]
        nembed = embeddings.shape[2]

        # --! encode attention over the sequence of embedded values
        #
        # --! embeddings, which are shaped as [B, T / kernzise, nembed] are encoded into attention with the same shape
        #
        # --! attention is produced for each sequence step (rows of attention matrix); this information can be
        # --! used to derive linear time-varying matrices that locally adapt to changes in dynamics
        embeddings = self.enc_model_mean(embeddings)

        # --! note that we omit matrix transpose here, relying on training to figure it out
        #
        # --! note also that the neural network, which generates matrices, is already configured to
        # --! cover the entire prediction horizon, i.e. lookback and forecast windows
        model_i = torch.flatten(embeddings, start_dim=1)
        mat = self.model_mean(model_i).reshape(batsize, -1, nembed, nembed)

        # --! accumulate matrix products to enable predictions, such as z2 = A1*z1, z3 = A2*A1*z1, etc,
        # --! where zi are our embeddings, and where Ai are linear time-varying matrices
        mat = util_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial conditions of embedded values, i.e. the first slice
        #
        # --! the initial conditions (ic) are shaped as [B, 1, 1, nfun] to allow tensor broadcasting
        # --! when multiplying by dynamics matrices
        embed_ic = torch.unsqueeze(embeddings[:, :1], -2)

        # --! predict the stationary evolution of embedded values by multiplying their initial conditions
        # --! by powered dynamics matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nembed], i.e.
        # --! batches with functions trajectories consisting of inidividual function-points [1, nembed]
        embed_pred = torch.matmul(embed_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        embed_pred = torch.squeeze(embed_pred, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(embed_pred, [embed_nsample - 1, self.embed_nsample_fore], dim=1)

    def predict_zeta(self, embeddings):

        # --! constants for convenience
        batsize = embeddings.shape[0]
        embed_nsample = embeddings.shape[1]
        nembed = embeddings.shape[2]

        # --! based on history (a lookback window), generate a sequence of piecewise-linear matrices that
        # --! capture the evolution of function values
        model_i = torch.flatten(embeddings, start_dim=1)
        mat = self.model_zeta(model_i).reshape(batsize, -1, nembed, nembed)
        mat = util_nn.cumprod_mat(mat[:, 1:])

        # --! extract the initial condition of embedded history
        #
        # --! the initial condition (ic) is shaped as [B, 1, 1, nembed] to allow tensor broadcasting
        # --! when multiplying by the matrices
        embed_ic = torch.unsqueeze(embeddings[:, :1], -2)

        # --! predict the evolution of embeddings by multiplying their initial conditions by the matrices
        #
        # --! both tensors are broadcasted together and multiplied to produce a shape [B, H - 1, 1, nembed],
        # --! i.e. a batch with function trajectories consisting of individual zeta points [1, nembed]
        embed_pred = torch.matmul(embed_ic, mat)

        # --! remove extra singleton dimensions that were needed for the broadcasting of multiplication
        embed_pred = torch.squeeze(embed_pred, -2)

        # --! return separate predictions for lookback and forecast windows
        #
        # --! note that the lookback window is -1, because we do not predict the initial condition
        return torch.split(embed_pred, [embed_nsample - 1, self.embed_nsample_fore], dim=1)

    def forward(self, timeseries):

        # --! from given timeseries, embed mean latent values and then predict these embeddings
        # --! starting from the first value (initial condition) upto a specified horizon
        mean = self.embed_mean(timeseries)
        mean_pred_back, mean_pred_fore = self.predict_mean(mean)
        mean_pred_back = torch.cat([mean[:, :1], mean_pred_back], dim=1)

        # --! do a similar procedure to embedded zeta values
        zeta = self.embed_zeta(timeseries)
        zeta_pred_back, zeta_pred_fore = self.predict_zeta(zeta)
        zeta_pred_back = torch.cat([zeta[:, :1], zeta_pred_back], dim=1)

        # --! decode predicted mean embeddings to time series domain
        #
        # --! time series are decoded as slice rows shaped as [B, T / kernsize, kernsize],
        # --! so we reshape the decoded results back to their
        # --! original shape [B, T, ndim]
        ts_mean = self.dec_mean(torch.cat([mean_pred_back, mean_pred_fore], dim=1))
        ts_mean = ts_mean.reshape(ts_mean.shape[0], -1, self.ntarget)

        # --! decode predicted zeta embeddings to time series domain
        ts_zeta = self.dec_zeta(torch.cat([zeta_pred_back, zeta_pred_fore], dim=1))
        ts_zeta = ts_zeta.reshape(ts_zeta.shape[0], -1, self.ntarget * 2)

        return ts_mean, ts_zeta, mean, mean_pred_back, zeta, zeta_pred_back

    def freeze_mean(self):
        util_nn.freeze_module(self.enc_mean)
        if self.nfeature > 1:
            util_nn.freeze_module(self.prune_mean)
        util_nn.freeze_module(self.enc_model_mean)
        util_nn.freeze_module(self.model_mean)
        util_nn.freeze_module(self.dec_mean)

    def freeze_zeta(self):
        util_nn.freeze_module(self.enc_zeta)
        if self.nfeature > 1:
            util_nn.freeze_module(self.prune_zeta)
        util_nn.freeze_module(self.model_zeta)
        util_nn.freeze_module(self.dec_zeta)

    def unfreeze(self):
        util_nn.unfreeze_module(self.enc_mean)
        util_nn.unfreeze_module(self.enc_zeta)
        if self.nfeature > 1:
            util_nn.unfreeze_module(self.prune_mean)
            util_nn.unfreeze_module(self.prune_zeta)
        util_nn.unfreeze_module(self.enc_model_mean)
        util_nn.unfreeze_module(self.model_mean)
        util_nn.unfreeze_module(self.model_zeta)
        util_nn.unfreeze_module(self.dec_mean)
        util_nn.unfreeze_module(self.dec_zeta)

