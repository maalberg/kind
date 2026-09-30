# --! cavity example: function and class definitions --!


from abc import abstractmethod
from abc import ABC as interface

import torch
import numpy as np

import reinforcement_learning


class environment(reinforcement_learning.environment):
    def __init__(self, args, reward_fn):

        # --! transform mech mode representation to respect llrf library interface
        mech_mode = {
            'f': args.mech_f,
            'Q': args.mech_q,
            'K': args.mech_k
        }

        # --! discrete-time matrices of mechanical modes
        status, am, bm, cm, dm = cav_ss_mech(mech_mode)
        status, self.am, self.bm, self.cm, self.dm, _ = ss_discrete(am, bm, cm, dm, Ts=args.step_t)

        self.cav_wh = np.pi * args.gen_f / args.cav_q   # half bandwidth, rad/s
        cav_rl = 0.5 * args.cav_r * args.cav_q     # loaded resistance (linac convention), Ohm

        # --! prepare arrays for complex cavity signals
        self.sig_vf  = np.zeros(args.step_n, dtype=complex) # forward voltage signal
        self.sig_vb  = np.zeros(args.step_n, dtype=complex) # beam voltage signal

        beam_start = args.fill_n
        beam_end = beam_start + args.beam_n

        # --! define constant shapes of complex cavity signals
        self.sig_vf[:]                   =  cav_rl * args.gen_i  * 0.1   # define forward voltage
        self.sig_vb[beam_start:beam_end] = -cav_rl * args.beam_i * 0.1   # define beam voltage

        self.gen_i = args.gen_i
        self.beam_i = args.beam_i

        # --! states
        self.state_cav = 0 + 0 * 1j # state of cavity equation, complex
        self.state_mech = np.matrix(np.zeros(self.bm.shape)) # state of mechanical equation, real

        # --! observables
        self.obs_dw = 0.0  # instantaneous detuning

        # --! time step configuration
        self.step_cnt = 0
        self.step_t = args.step_t
        self.nstep = args.step_n

        self.r_fn = reward_fn

    def reset(self, ic=0.0):

        # --! reset states
        self.state_cav = ic
        self.state_mech = np.matrix(np.zeros(self.bm.shape))

        # --! reset step counter
        self.step_cnt = 0

        # --! reset instantaneous detuning and return it as an observable
        self.obs_dw = 0.0
        return self.state_cav, self.obs_dw

    def step(self, action):

        # --! generate current microphonics
        dw_micr = 2.0 * np.pi * np.random.randn() * 0.2

        i = self.step_cnt

        status, self.state_cav, vr, self.obs_dw, self.state_mech = sim_scav_step(
            self.cav_wh,
            self.obs_dw,
            action + dw_micr,
            self.sig_vf[i],
            self.sig_vb[i],
            self.state_cav,
            self.step_t,
            beta      = 1e4,
            state_m0  = self.state_mech,
            Am        = self.am,
            Bm        = self.bm,
            Cm        = self.cm,
            Dm        = self.dm,
            mech_exe  = True)

        # --! update step counting
        self.step_cnt += 1
        done = self.step_cnt == self.nstep

        vc = self.state_cav if np.isscalar(self.state_cav) else self.state_cav[0,0]

        return vc, self.obs_dw[0, 0], None, done

    def replay(self, ic, policy, obs_nsample=1, skip_nsample=0):
        pass

    @property
    def reward_fn(self):
        return self.r_fn

