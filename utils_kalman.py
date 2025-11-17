import numpy as np
import scipy as sp
from filterpy.kalman import KalmanFilter as KF

import utils_detuning

def filter_detuning(param):
    """ Creates a linear Kalman filter to filter a cavity detuning signal. """

    f = param['f']
    q = param['q']
    k = param['k']
    dt = param['dt']
    kalman_q = param['kalman_q']
    kalman_r = param['kalman_r']

    # --! create observed process matrices
    a = utils_detuning.make_mm_a_array(f, q)
    b = utils_detuning.make_mm_b_array(f, k)
    c = utils_detuning.make_mm_c_array(len(f))

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
