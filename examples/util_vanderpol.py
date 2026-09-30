# --! van der pol utilities --!


import numpy as np
from scipy.integrate import solve_ivp


def step_vanderpol(s, damping, **kwargs):

    # --! specify default values of optional parameters
    options = {
        'dt'        : 1e-6,
    }

    # --! overwrite those defaults which were actually provided by user
    options.update(kwargs)

    # --! for convinience, extract optional parameters from dictionary
    dt = options.get('dt')

    # --! derive solver timing based on user time step
    bt = (0.0, dt)                                     # time boundaries

    # --! assemble pendulum parameters that are passed into solver in addition to current state
    param = {
        'dv'   : damping,
    }

    # --! solve initial value problem
    ivp_solution = solve_ivp(vanderpol_update, bt, s, args=(param,))

    # --! as solved state, take the last entry from solution list shaped as [n, t],
    # --! where n and t are the number of states and time steps, respectively
    s = ivp_solution.y[:, -1]

    return s


def vanderpol_update(t, s, param):

    # --! get parameters
    dv = param.get('dv') # van der pol damping

    # --! van der pol features two states: position x and velocity y
    x, y = s

    # --! compute derivatives
    dx = y
    dy = dv * (1 - x ** 2) * y - x

    # --! return flattened derivatives of states
    return np.array([dx, dy])

