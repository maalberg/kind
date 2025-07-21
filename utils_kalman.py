import numpy as np
import scipy as sp
from filterpy.kalman import KalmanFilter as KF

def filter_detuning(param):
    """Creates a linear Kalman filter to filter a cavity detuning signal.
    """
    dt    = param['dt']
    omega = 2 * np.pi * param['f']
    q     = param['q']
    k     = 2 * np.pi * param['k']
    a     = np.array([[0, 1], [-omega**2, -omega/q]])
    b     = np.array([[0], [-k*omega**2]])

    kf    = KF(dim_x=2, dim_z=1)
    kf.F  = sp.linalg.expm(a * dt)
    kf.H  = np.array([[1, 0]])
    kf.B  = b * dt

    # initial state and covariances
    kf.x  = np.array([[0.], [0.]])
    kf.P *= np.eye(2)   # uncertainty about the initial condition

    # --! here is the point: we assume the measurement is noisy, which is not exactly true (the sensor improved),
    # --! and we overestimate our model knowledge 
    kf.R *= 0.1  # measurement noise
    kf.Q *= 0.01  # process uncertainty

    return kf