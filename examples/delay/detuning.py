# --! example: function and class definitions for detuning

import numpy as np
import torch
import control as ct

from collections import namedtuple, deque
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp

import reinforcement_learning as rl

import matplotlib.pyplot as plt


class detuning(rl.environment):

    def __init__(self, cavity_param, reward_fn, dt_dynamics=1e-4, dt_control=1e-2, t_end=1):

        self.cavity_param = cavity_param
        self.r_fn = reward_fn

        # --! number of cavity states includes two rf states and the number of mechaical modes times two
        nstate = 2 + 2 * len(cavity_param.get('mm_f'))
        self.state = np.zeros(nstate)

        self.dt_dynamics = dt_dynamics
        self.dt_control = dt_control

        self.step_cnt = 0
        self.nstep = t_end // self.dt_control

    def reset(self, ic):

        # --! reset step counter
        self.step_cnt = 0

        # --! reset the state to initial condition
        self.state = ic

        # --! return detuning
        return self._sum_detuning()

    def step(self, action):

        # --! based on current state and action, calculate reward
        reward = self.reward_fn(np.array([self.state[2:]]), np.array([[action]]))

        # --! define step timing
        t_start = self.step_cnt * self.dt_control
        t = t_start + np.arange(0, self.dt_control, self.dt_dynamics)
        t_span = (t[0], t[-1])

        # --! solve initial value problem
        solution = solve_ivp(
            cavity_update, t_span, self.state,
            t_eval=t, args=(action, self.cavity_param))

        # --! save last integrated value as next state
        self.state = solution.y[:, -1]

        self.step_cnt = self.step_cnt + 1
        done = self.step_cnt==self.nstep

        return self._sum_detuning(), reward, done

    @property
    def reward_fn(self):
        return self.r_fn

    def _sum_detuning(self):

        # --! skip the first two rf states to extract only mechanical part
        state_mech = self.state[2:]
        nmode = len(state_mech) // 2

        # --! sum mechanical modes' positions to derive detuning
        detuning = np.sum([state_mech[2 * j] for j in range(nmode)])

        return np.array([[detuning]])


class detuning_adapter(rl.environment):

    def __init__(self, env):
        self.env = env
        self.r_fn = detuning_reward_adapter(env.reward_fn)

    def reset(self, ic):
        ic = ic.detach().cpu().numpy()
        state = self.env.reset(np.squeeze(ic))
        return torch.from_numpy(state).to(dtype=torch.float32)

    def step(self, action):
        action = action.detach().cpu().numpy()

        next_state, reward, done = self.env.step(np.squeeze(action))

        next_state = torch.from_numpy(next_state).to(dtype=torch.float32)
        reward = torch.tensor(reward).to(dtype=torch.float32)
        done = torch.tensor(done)

        return next_state, reward, done

    @property
    def reward_fn(self):
        return self.r_fn


class detuning_reward:
    def __init__(self, q, r, setpoint):
        self.q = np.atleast_2d(q)
        self.r = np.atleast_2d(r)
        self.setpoint = np.atleast_2d(setpoint)

    def __call__(self, state, action):
        x_err = state - self.setpoint

        state_cost = x_err @ self.q @ x_err.T
        action_cost = action @ self.r @ action.T

        return -(state_cost + action_cost)


class detuning_reward_adapter:
    def __init__(self, reward_fn, dtype=torch.float32, device=None):

        self.q = torch.atleast_2d(torch.as_tensor(reward_fn.q, dtype=dtype, device=device))
        self.r = torch.atleast_2d(torch.as_tensor(reward_fn.r, dtype=dtype, device=device))
        self.setpoint = torch.atleast_2d(torch.as_tensor(reward_fn.setpoint, dtype=dtype, device=device))

    def __call__(self, state, action):

        state = torch.atleast_2d(state)
        action = torch.atleast_2d(action)

        state_err = state - self.setpoint

        state_cost = state_err @ self.q @ state_err.transpose(-1, -2)
        action_cost = action @ self.r @ action.transpose(-1, -2)

        return -(state_cost + action_cost)


class base_policy:
    def __init__(self, est_a, est_b, reg_k, setpoint, dtype=torch.float32, device=None):

        self.est_a = torch.atleast_2d(torch.as_tensor(est_a, dtype=dtype, device=device))
        self.est_b = torch.atleast_2d(torch.as_tensor(est_b, dtype=dtype, device=device))
        self.reg_k = torch.atleast_2d(torch.as_tensor(reg_k, dtype=dtype, device=device))

        self.setpoint = torch.atleast_2d(torch.as_tensor(setpoint, dtype=dtype, device=device))

        self.est_state = torch.zeros(1, est_a.shape[0])
        self.action = torch.zeros(1, 1)

    def __call__(self, obs):
        obs_err = obs - self.setpoint

        # --! prepare current action as a combination of current observation and past action
        action = torch.cat([obs_err, self.action], dim=-1)

        # --! estimate full state
        est_ax = torch.matmul(self.est_state, torch.transpose(self.est_a, -1, -2))
        est_bu = torch.matmul(action, torch.transpose(self.est_b, -1, -2))
        est_state = est_ax + est_bu

        # --! with full state estimated, use full-state feedback to get action
        action = -torch.matmul(est_state, torch.transpose(self.reg_k, -1, -2))

        # --! save estimated state and action for next iteration
        self.est_state = est_state
        self.action = action

        return action


def make_base_policy(
    f, q, k,
    dt,
    setpoint=0.0,
    state_cost=[1.0, 0.1], state_max=[1.0, 100.0],
    action_cost=10.0, action_max=1.0, est_q=1.0, est_r=1.0):
    """Makes an LQG regulator."""

    plant = make_mm_system(f=f, q=q, k=k)
    plant = ct.c2d(plant, dt)

    state_max = np.diag(np.tile(state_max, len(f)))

    # --! normalize plant matrices
    a = np.linalg.inv(state_max) @ plant.A @ state_max
    b = np.linalg.inv(state_max) @ plant.B * action_max

    reg_k = make_reg(a, b, state_cost, state_max, action_cost)
    est_a, est_b = make_est(plant, q=est_q, r=est_r)

    return base_policy(est_a, est_b, reg_k, setpoint)


class replay_factory(rl.replay_factory):

    obs_ndim = 1
    action_ndim = 1
    mask_ndim = 1

    def __init__(self):
        pass

    def create(self, env, env_ic, policy, back_nsample, skip_nsample=0):

        # --! reset environment to begin replay from initial condition
        s = env.reset(env_ic)

        # --! done flag for sanity checks
        done = False

        print(f'>>> replay factory: skipping {skip_nsample} samples')

        # --! skip specified number of samples
        for k in range(skip_nsample):
            a = policy.base(s)

            # --! provided residual policy is available, add residual action to the base one
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            s, reward, done = env.step(a)

            if done: break

        if done: return None

        # --! make a window for recent states and actions
        #
        # --! a deque allows to push new data in,
        # --! which automatically pops out old data once deque's capacity is filled
        sa_window = deque(maxlen=back_nsample)

        # --! fill the first window
        for _ in range(back_nsample):
            a = policy.base(s)

            # --! provided residual policy is available, add residual action to the base one
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            sa_window.append((s, a))
            s, reward, done = env.step(a)

            if done: break

        # --! make an empty replay buffer
        buf = rl.replay()

        # --! fill replay with back windows, rewards, etc.
        while not done:

            # --! encode state at time t from a window of recent observations
            state = self.encode_state(sa_window)

            # --! based on observation at time t + 1, compute action at time t + 1
            a = policy.base(s)

            # --! provided residual policy is available, add residual action to the base one
            if policy.residual is not None:
                a = a + torch.squeeze(policy.residual(s), 0)

            # --! update window with observation and action at time t + 1, and encode next state
            sa_window.append((s, a))
            next_state = self.encode_state(sa_window)

            # --! replay buffer receives:
            #
            # --! state at time t
            # --! reward at time t
            # --! state at time t + 1
            # --! done flag at time t
            buf.add(
                state,
                reward,
                next_state,
                done
            )

            # --! make environment step to get next observations, rewards, etc.
            s, reward, done = env.step(a)

        return buf

    def encode_state(self, sa_window):

        # --! unpack given deque into sa tuples, and then zip all s's together and all a's together
        s, a = zip(*sa_window)

        # --! concatenate zipped s tuples into torch tensor, do the same for a
        s = torch.cat(s, dim=0)
        a = torch.cat(a, dim=0)

        # --! concatenate s, a and mask tensors and return result as a 3D tensor
        return torch.unsqueeze(torch.cat([s, a, torch.ones_like(a)], dim=-1), 0)

    def extract_current_s(self, state):
        pass

    def extract_current_a(self, state):
        pass

    def update_current_a(self, state, a):
        pass

    def update_state(self, state, s):
        pass
    

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


def make_mm_system(f=np.array([100.0]), q=np.array([1000.0]), k=np.array([1.0])):
    a = make_mm_a_array(f, q)
    b = make_mm_b_array(f, k)
    c = make_mm_c_array(len(f))
    d = np.array(([[0]]))

    return ct.ss(a, b, c, d)


def make_reg(a, b, state_cost=[1.0, 0.1], state_max=np.array([[1.0, 0.0], [0.0, 100.0]]), action_cost=1.0):

    q = np.diag(np.tile(state_cost, a.shape[0] // 2))
    r = np.diag(np.atleast_1d(action_cost))

    k, _, _ = ct.dlqr(a, b, q, r)

    return k @ np.linalg.inv(state_max)


def make_est(plant, q=1.0, r=1.0):

    q = np.eye(1) * q
    r = np.eye(1) * r

    # --! make Q and R symmetric
    q = q @ q.T
    r = r @ r.T

    est_gain, _, _ = ct.dlqe(plant, q, r)
    est_a = plant.A - est_gain @ plant.C
    est_b = np.hstack([est_gain, plant.B])  # input: [y, u]

    return est_a, est_b


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

