# --! care: methods to simulate cavity resonance --!


import numpy as np
from scipy.integrate import solve_ivp


def step_cav(vc, vf, sm, me, mm, **kwargs):
    """
    Simulates cavity response by solving cavity equations in continuous time.

    Required parameters:
        vc: complex, cavity voltage, V
        vf: complex, forward voltage, V
        sm: numpy column-like array, mechanical state
        me: tuple of numpy arrays, electrical matrices a and b, continuous-time
        mm: tuple of numpy arrays, mechanical matrices a, b, c and d, continuous-time

    Optional parameters:
        vb:        complex, beam voltage, V [default=0]
        a:         float, tuner action, rad/s [default=0]
        dt:        float, time step, s [default=1e-6]
        beta:      float, input coupling factor, unitless [default=1e4]

    Returns:
        vc: complex, cavity voltage, V
        dc: float, cavity detuning, rad/s
        sm: numpy column vector, mechanical state
    """

    # --! specify default values of optional parameters
    options = {
        'vb'        : 0.0 + 1j * 0.0,
        'a'         : 0.0,
        'dt'        : 1e-6,
        'beta'      : 1e4,
    }

    # --! overwrite those defaults which were actually provided by user
    options.update(kwargs)

    # --! for convinience, extract optional parameters from dictionary
    vb = options.get('vb')
    a = options.get('a')
    dt = options.get('dt')
    beta = options.get('beta')

    # --! derive solver timing based on user time step
    bt = (0.0, dt)                                     # time boundaries

    # --! format cavity state (electrical and mechanical) as a flattened numpy array for solver
    s = np.concatenate([[np.real(vc), np.imag(vc)], sm.flatten()])

    # --! assemble cavity parameters that are passed into solver in addition to current state
    cav_param = {
        'ae'   : me.a,
        'be'   : me.b,

        'am'   : mm.a,
        'bm'   : mm.b,
        'cm'   : mm.c,

        'vf'   : vf,
        'vb'   : vb,

        'a'    : a,

        'beta' : beta,
    }

    # --! solve initial value problem
    ivp_solution = solve_ivp(cav_update, bt, s, args=(cav_param,))

    # --! as solved state, take the last entry from solution list shaped as [n, t],
    # --! where n and t are the number of states and time steps, respectively
    s = ivp_solution.y[:, -1]

    # --! decompose solved state into complex cavity voltage and mechanical state
    vc = s[0] + 1j * s[1]
    sm = np.array(s[2:]).reshape((-1, 1))

    # --! compute resulting cavity detuning
    dc = comp_detuning(sm, mm.c, a)

    return vc, dc, sm


def cav_update(t, s, param):

    # --! get electrical matrices
    ae = param.get('ae')
    be = param.get('be')

    # --! get mechanical matrices
    am = param.get('am')
    bm = param.get('bm')
    cm = param.get('cm')

    # --! get complex voltages
    vf = param.get('vf')
    vb = param.get('vb')

    # --! get tuner action
    a  = param.get('a')

    # --! get beta
    beta = param.get('beta')

    # --! extract electrical and mechanical cavity states as numpy column vectors
    se = np.array(s[:2]).reshape((-1, 1))
    sm = np.array(s[2:]).reshape((-1, 1))

    # --! update forward voltage using beta coupling factor
    vf_beta = vf * beta / (beta + 1)

    # --! forward and beam voltages drive electrical equation
    ue_vf = np.array([np.real(vf_beta), np.imag(vf_beta)]).reshape((-1, 1))
    ue_vb = np.array([np.real(vb), np.imag(vb)]).reshape((-1, 1))

    # --! compute cavity detuning to update electrical matrix A
    dc = comp_detuning(sm, cm, a)

    # --! update electrical matrix A with cavity detuning
    ae[0, 1] = -dc
    ae[1, 0] =  dc

    # --! cavity gradient in megavolts squared drives mechanical equation
    um = np.square(np.sqrt(np.square(se[0]) + np.square(se[1])) * 1e-6)

    # --! compute electrical and mechanical derivatives
    #
    # --! note that electrical equation applies factor 2 on forward and beam voltages
    # --! to account for current passing through input coupler
    de = ae @ se + 2 * be @ (ue_vf + ue_vb)
    dm = am @ sm + bm * um

    # --! return flattened derivatives of states
    return np.concatenate([de.flatten(), dm.flatten()])


def comp_detuning(sm, cm, a):
    """
    Computes detuning.
        
    Parameters:
        sm: numpy column vector, mechanical state
        cm: numpy matrix, mechanical output matrix c
        a:  float, action, rad/s

    Returns:
        dc: float, cavity detuning, rad/s
    """

    dc = cm @ sm + a
    return np.squeeze(dc)

