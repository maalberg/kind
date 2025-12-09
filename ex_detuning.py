# --! example: function and class definitions for detuning

import os

import numpy as np
import scipy as sp

import torch

import control as ct
from filterpy.kalman import KalmanFilter as KF

from collections import namedtuple
from scipy.linalg import block_diag

import util_data


# --! mechanical mode properties --!
mechanical_mode = namedtuple('mechanical_mode', 'f q k t')


def make_rf_a(f, q):
    w     = 2 * np.pi * f
    hbw   = w/2/q  # < half-bandwidth of an rf cavity in rad/s

    return np.array([
        [-hbw,     0.],
        [ 0.,    -hbw],
    ])


def make_rf_b(f, q):
    w     = 2 * np.pi * f
    hbw   = w/2/q  # < half-bandwidth of an rf cavity in rad/s

    return np.array([
        [hbw,    0.],
        [0.,    hbw],
    ])


def make_mm_a(f, q):
    """Constructs the matrix A of a single mechanical mode (mm)."""
    w = 2 * np.pi * f
    return np.array([
        [ 0,             1  ],
        [-np.square(w), -w/q],
    ])


def make_mm_b(f, k):
    """Constructs the matrix B of a single mechanical mode (mm)."""
    w = 2 * np.pi * f
    return np.array([
        [0               ],
        [k * np.square(w)],
    ])


def make_mm_c():
    return np.array([[1, 0]])


def make_mm_a_array(f, q):
    return block_diag(*[make_mm_a(f, q) for f, q in zip(f, q)])


def make_mm_b_array(f, k):
    return np.concatenate([make_mm_b(f, k) for f, k in zip(f, k)], axis=0)


def make_mm_c_array(nf=1):
    return np.tile(make_mm_c(), nf)


def make_mm_system(f=np.array([100.]), q=np.array([1000.]), k=np.array([1.]), dt=0.):
    a = make_mm_a_array(f, q)
    b = make_mm_b_array(f, k)
    c = make_mm_c_array(len(f))
    d = np.array(([[0]]))

    return ct.ss(a, b, c, d, dt, name='mm_plant')


def make_lqr(f, q, k, action_cost=1., state_cost=1., dt=0.):

    lqr_plant = make_mm_system(f=f, q=q, k=k, dt=dt)

    # --! define maximum expected signal values
    det_max = 1.0      # detuning max [rad/s]
    vel_max = 200.0    # velocity max [rad/s^2]
    lqr_max = 1.0     # control max [V]
    x_norm = np.diag(np.tile([det_max, vel_max], len(f)))
    u_norm = lqr_max

    # --! normalize plant matrices
    a_norm = np.linalg.inv(x_norm) @ lqr_plant.A @ x_norm
    b_norm = np.linalg.inv(x_norm) @ lqr_plant.B * u_norm

    q_norm = np.diag(np.tile([state_cost, 0.1], len(f)))
    r_norm = np.diag([action_cost])

    k_norm, _, _ = ct.lqr(a_norm, b_norm, q_norm, r_norm)

    return lqr_plant, k_norm @ np.linalg.inv(x_norm)


def make_est(lqr_plant, q=0.1, r=0.1):

    q = np.eye(1) * q
    r = np.eye(1) * r

    # --! make Q and R symmetric
    q = q @ q.T
    r = r @ r.T

    est_gain, _, _ = ct.lqe(lqr_plant, q, r)
    est_a = lqr_plant.A - est_gain @ lqr_plant.C
    est_b = np.hstack([est_gain, lqr_plant.B])  # input: [y, u]

    return est_a, est_b


def cavity_update(t, x, u, param):

    # --! get parameters
    rf_f = param.get('rf_f')
    rf_q = param.get('rf_q')
    rf_v = param.get('rf_v')
    rf_len = param.get('rf_len')
    mm_control_used = param.get('mm_control_used')
    mm_f = param.get('mm_f')
    mm_q = param.get('mm_q')
    nmm = len(mm_f)
    mm_k = param.get('mm_k')
    mm_t = param.get('mm_t')

    # --! extract current state of a cavity field: real and imaginary components
    rf_x = np.array(x[:2]).reshape((-1, 1))

    # --! extract current states of all mechanical modes: displacements and velocities
    mm_x = np.array(x[2:]).reshape((-1, 1))

    # --! assemble input to cavity field: real and imaginary parts of a generator voltage
    rf_u = np.array([
        [rf_v[0]],
        [rf_v[1]]
    ])

    # --! compute input to mechanical mode: accelerating field gradient squared
    #
    # --! field gradient has units MV/m, but since we simulate only one cell,
    # --! and the length of one cell is passed as a parameter,
    # --! we need to adjust the total gradient
    rf_grad = np.sqrt(np.square(rf_x[0]) + np.square(rf_x[1]))
    rf_grad = rf_grad * rf_len
    rf_grad = np.square(rf_grad)

    # --! create rf matrices A and B
    rf_a = make_rf_a(rf_f, rf_q)
    rf_b = make_rf_b(rf_f, rf_q)

    # --! update detuning in rf matrix A
    mm_disp = np.sum([mm_x[2*j] for j in range(nmm)])
    rf_a[0, 1] = -mm_disp
    rf_a[1, 0] =  mm_disp

    # --! assemble mechanical mode matrices: A and B
    mm_a = block_diag(*[make_mm_a(f, q) for f, q in zip(mm_f, mm_q)])
    mm_b_field = np.concatenate([make_mm_b(f, k) for f, k in zip(mm_f, mm_k)], axis=0)
    mm_b_control = np.concatenate([make_mm_b(f, k) for f, k in zip(mm_f, mm_k)], axis=0)

    # --! split matrix B of mechanical modes into per-mode B matrices
    mm_b_mode = np.split(mm_b_field, mm_b_field.shape[0] // 2, axis=0)

    # --! split mechanical time boundary array into per-mode parts
    mm_t_mode = np.split(mm_t, mm_t.shape[0], axis=0) # split into rows

    for mat, timespan in zip(mm_b_mode, mm_t_mode):
        if not (timespan[0, 0] <= t and t < timespan[0, 1]):
            mat[:] = 0.

    # --! mechanical modes are excited by field gradient ...
    mm_u_field = rf_grad

    mm_u_control = 0.
    if mm_control_used:
        # --! ... and compensated by control (if used)
        mm_u_control = u

    # --! calculate derivatives
    rf_dx = rf_a @ rf_x + rf_b @ rf_u
    mm_dx = mm_a @ mm_x + mm_b_field * mm_u_field + mm_b_control * mm_u_control

    return np.array([
        *rf_dx.flatten(),
        *mm_dx.flatten(),
    ])


def cavity_output(t, x, u, param):
    """ Outputs summed positions of all mechanical modes, i.e. cavity detuning. """
    mm_x = x[2:]
    nmm  = len(mm_x) // 2
    mm_d = np.sum([mm_x[2*j] for j in range(nmm)])
    return np.array([mm_d])


def estimator_update(t, x, u, param):

    est_a = param.get('est_a')
    est_b = param.get('est_b')
    est_x = np.array(x).reshape((-1, 1))
    est_u = np.array(u).reshape((-1, 1))

    est_dx = est_a @ est_x + est_b @ est_u

    return np.array([
        *est_dx.flatten(),
    ])


def estimator_output(t, x, u, param):
    return np.array([x])


def control_output(t, x, u, param):

    # --! get parameters
    lqr_gain = param.get('lqr_gain')

    est_x = np.array(u).reshape((-1, 1))
    return -(lqr_gain @ est_x)


def sim_cavity_control(t,
                       start_jsample=0, end_jsample=200,
                       lqr_used=False, lqr_q=1., lqr_r=1.,
                       est_q=1., est_r=1.,
                       mm_f=np.array([40.]), mm_q=np.array([400.]), mm_k=np.array([1.]), mm_t=np.array([[-1., -1.]]),
                       control_f=np.array([40.]), control_q=np.array([400.]), control_k=np.array([1.]),
                       plotted=False,
                       solve_method='RK45'):
    """ Simulates cavity equations under control. """

    # --! actualize time boundaries
    for timespan in np.split(mm_t, mm_t.shape[0], axis=0): # split into rows
            if timespan[0, 0]==-1.:
                timespan[0, 0] = t[0]
            if timespan[0, 1]==-1.:
                timespan[0, 1] = t[-1]

    # --! prepare parameters for cavity plant simulation
    cavity_param = {
        'rf_f' : 1.3e9,
        'rf_q' : 4e6,
        'rf_v' : [9.5, 0.],
        'rf_len' : 0.1615,
        'mm_control_used' : lqr_used,
        'mm_f' : mm_f,
        'mm_q' : mm_q,
        'mm_k' : mm_k,
        'mm_t' : mm_t,
    }

    # --! number of cavity states includes two rf states and the number of mechaical modes times two
    nstate = 2 + 2 * len(cavity_param.get('mm_f'))

    # --! wrap a cavity plant in a nonlinear input/output system
    cavity = ct.nlsys(
        cavity_update, cavity_output,
        states=nstate,
        name='cavity',
        inputs=1, outputs=1,
        params=cavity_param)

    # --! create cavity control
    lqr_plant, lqr_gain = make_lqr(f=control_f, q=control_q, k=control_k,
                                   state_cost=lqr_q, action_cost=lqr_r)

    est_a, est_b = make_est(lqr_plant, q=est_q, r=est_r)
    estimator_param = {
        'est_a': est_a,
        'est_b': est_b,
    }
    est_plant_nstate = lqr_plant.nstates
    estimator = ct.nlsys(
        estimator_update, estimator_output,
        name='estimator',
        states=est_plant_nstate,
        inputs=2, outputs=est_plant_nstate,
        params=estimator_param
    )

    control_param = {
        'lqr_gain': lqr_gain,
    }
    control = ct.nlsys(
        None, control_output,
        name='control',
        inputs=est_plant_nstate, outputs=1,
        params=control_param
    )

    # --! build a closed loop system
    cavity_closed = ct.interconnect(
        [control, estimator, cavity],
        connections=[
            ['cavity.u', 'control.y'],
            ['estimator.u[0]', 'cavity.y'],
            ['estimator.u[1]', 'control.y'],
            ['control.u', 'estimator.y'],
        ],
        outlist=['cavity.y', 'control.y'],
        outputs=['dw', 'pzt'],
    )

    # --! display input-output response
    resp = ct.input_output_response(cavity_closed, t, solve_ivp_method=solve_method)
    if plotted:
        resp.plot(plot_inputs=False)

    timeseries_nsample = end_jsample - start_jsample#resp.outputs[0].shape[0] - skip_nsample

    return np.concatenate(
        [
            resp.outputs[0][start_jsample:end_jsample].reshape(-1, 1),
            resp.outputs[1][start_jsample:end_jsample].reshape(-1, 1),
        ], axis=1).reshape(-1, timeseries_nsample, 2)


def filter_detuning(param):
    """ Creates a linear Kalman filter to filter a cavity detuning signal. """

    f = param['f']
    q = param['q']
    k = param['k']
    dt = param['dt']
    kalman_q = param['kalman_q']
    kalman_r = param['kalman_r']

    # --! create observed process matrices
    a = make_mm_a_array(f, q)
    b = make_mm_b_array(f, k)
    c = make_mm_c_array(len(f))

    # --! there two states (position and velocity) per one mechanical mode frequency
    nstate = 2 * len(f)
    nmeas = 1

    # --! create a linear Kalman filter
    kf = KF(dim_x=nstate, dim_z=nmeas)
    kf.F = sp.linalg.expm(a * dt)
    kf.B = b * dt
    kf.H = c

    # initial state and covariances
    kf.x = np.zeros((nstate, 1))
    kf.P *= 10. * np.eye(nstate)   # uncertainty about the initial condition

    kf.R *= kalman_r  # measurement noise
    kf.Q *= kalman_q  # process uncertainty

    return kf


class dataset_synthetic(util_data.dataset):
    """Models a synthetic detuning database for nominal filtering."""

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='nom'):
        file_name = self.file_name + '_nom' + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        return window[:, :, [0]]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def init_normalization(self, timeseries):

        detuning_control, mask = torch.split(timeseries, [2, 1], dim=-1)

        # --! take global statistics
        self.mean = detuning_control.mean()
        self.std = detuning_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==1:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = torch.clip((detuning - self.mean) / self.std, min=-3.0, max=3.0)
            control = torch.clip((control - self.mean) / self.std, min=-3.0, max=3.0)
            control = control * mask

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==1:
            window = window * self.std + self.mean
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = detuning * self.std + self.mean
            control = control * self.std + self.mean

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def noise(self, window):
        detuning, control, mask = torch.split(window, 1, dim=-1)
        detuning = detuning + 1e-3 * torch.randn(detuning.shape)

        return torch.cat([detuning, control, mask], dim=-1)


class dataset_measured(util_data.dataset):
    """Models measured detuning dataset, unfiltered."""

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='nom'):
        file_name = self.file_name + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        return window[:, :, [0]]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def init_normalization(self, timeseries):

        detuning_control, mask = torch.split(timeseries, [2, 1], dim=-1)

        # --! take global statistics
        self.mean = detuning_control.mean()
        self.std = detuning_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==1:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = torch.clip((detuning - self.mean) / self.std, min=-3.0, max=3.0)
            control = torch.clip((control - self.mean) / self.std, min=-3.0, max=3.0)
            control = control * mask

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==1:
            window = window * self.std + self.mean
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = detuning * self.std + self.mean
            control = control * self.std + self.mean

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def noise(self, window):
        return window


class dataset_filtered(util_data.dataset):
    """Models measured detuning dataset, filtered into nominal and excursion partitions."""

    # --! minimum standard deviation to avoid division by zero-like deviation values
    min_std = torch.tensor(1e-3, dtype=torch.float32)

    def __init__(self,
                 file_dir, file_name, file_ext,
                 data_nsample,
                 data_split_size,
                 batch_size, window_nsample):
        super().__init__(file_dir, file_name, file_ext,
                         data_nsample, data_split_size,
                         batch_size, window_nsample)

        self.mean = 0.
        self.std = self.min_std

    def make_path(self, data_type='nom'):
        file_name = self.file_name + '_' + data_type + self.file_ext
        return os.path.join(self.file_dir, file_name)

    def extract_target(self, window):
        return window[:, :, [0]]

    def adapt_mask(self, window):

        # --! mask stays as it is
        return window

    def extract_window(self, timeseries):

        # --! given time series should already contain windows
        return timeseries

    def init_normalization(self, timeseries):

        detuning_control, mask = torch.split(timeseries, [2, 1], dim=-1)

        # --! take global statistics
        self.mean = detuning_control.mean()
        self.std = detuning_control.std()
        self.std = torch.maximum(self.std, self.min_std)

    def normalize(self, window):

        if window.shape[-1]==1:
            window = torch.clip((window - self.mean) / self.std, min=-3.0, max=3.0)
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = torch.clip((detuning - self.mean) / self.std, min=-3.0, max=3.0)
            control = torch.clip((control - self.mean) / self.std, min=-3.0, max=3.0)
            control = control * mask

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def denormalize(self, window):

        if window.shape[-1]==1:
            window = window * self.std + self.mean
        else:
            detuning, control, mask = torch.split(window, 1, dim=-1)

            detuning = detuning * self.std + self.mean
            control = control * self.std + self.mean

            window = torch.cat([detuning, control, mask], dim=-1)

        return window

    def noise(self, window):

        # --! measured data already has noise
        return window

