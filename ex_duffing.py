# --! example: Duffing oscillator --!

import numpy as np
import scipy as sp

from scipy import signal
from scipy.integrate import solve_ivp

import util_dyna


def duffing_update(t, state, sim, u):
    """Generates a Duffing ``state`` update."""

    x1, x2 = state
    dx1 = x2
    dx2 = -sim.delta * x2 - sim.alpha * x1 - sim.beta * x1**3 + u + sim.gamma * np.cos(sim.omega * t)
    return [dx1, dx2]


class duffing(util_dyna.simulator):
    """Simulates a Duffing oscillator."""

    def __init__(self, beta, gamma, alpha=-1.0, delta=0.2, omega=1.0):

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega

        self.policy = None

    def simulate(self, ic, dt_control=1e-2, dt_sim=1e-4, t_final=100):

        x_buf = []
        dx_buf = []
        u_buf = []

        x_buf.append(ic[0])
        dx_buf.append(ic[1])
        u_buf.append(0.0) # << the first action is initialized with zero

        for step in range(int(t_final / dt_control)):

            # --! define the timing of local simulation
            t_local_start = step * dt_control
            t_local = t_local_start + np.arange(0, dt_control, dt_sim)
            t_span = (t_local[0], t_local[-1])

            # --! simulate locally
            solution = solve_ivp(duffing_update, t_span, [x_buf[-1], dx_buf[-1]], t_eval=t_local, args=(self, u_buf[-1]))

            # --! save the last integrated value from the local simulation
            x_buf.append(solution.y[0, -1])
            dx_buf.append(solution.y[1, -1])

            # --! based on integrated state, compute corresponding control input u
            state = [x_buf[-1], dx_buf[-1]]
            u = np.squeeze(self.policy(np.array(state).reshape((-1, 1))))

            u_buf.append(u)

        return x_buf, dx_buf, u_buf


def make_duffing(name):

    if name == 'ood':
        return duffing(20.0, 20.0)
    elif name == 'id':
        return duffing(1.0, 0.1)
    else:
        return None


def make_policy(duffing, q=[1.0, 0.1], r=[1.0]):
    """Makes a baseline LQR policy for a Duffing oscillator."""

    alpha = duffing.alpha
    delta = duffing.delta

    a = np.array([
        [ 0,      1    ],
        [-alpha, -delta],
    ])

    b = np.array([
        [ 0    ],
        [ 1    ],
    ])

    # --! discretize a continuous-time state-space system
    sys = signal.StateSpace(a, b, np.eye(2), np.zeros((2, 1)))
    sys = sys.to_discrete(1e-2)

    # --! take discrete matrices A and B
    a = sys.A
    b = sys.B

    p = sp.linalg.solve_discrete_are(a, b, np.diag(q), np.diag(r))

    # --! synthesize LQR gain matrix K = (B.T * P * B + R)^-1 * (B.T * P * A)
    bp = b.T.dot(p)
    lhs = bp.dot(b)
    lhs += r
    rhs = bp.dot(a)
    k = np.linalg.solve(lhs, rhs)

    # --! wrap the gain matrix in a callable policy strategy
    return util_dyna.lqr(k, noise=0.01)

