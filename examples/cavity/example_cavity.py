# --! cavity example: function and class definitions --!


import argparse
import torch
import numpy as np

from llrflibs.rf_sim import cav_ss_mech, sim_scav_step
from llrflibs.rf_control import ss_discrete

import reinforcement_learning


def create_args_parser():
    p = argparse.ArgumentParser(description='CAEN: radio frequency cavity environment')

    p.add_argument('-fg', '--gen_f', type=float, required=True, help='float, operating frequency, Hz')
    p.add_argument('-ig', '--gen_i', type=float, required=True, help='float, forward power equivalent current, A')

    p.add_argument('-rc', '--cav_r', type=float, required=True, help='float, R over Q of cavity, Ohm')
    p.add_argument('-qc', '--cav_q', type=float, required=True, help='float, loaded quality factor of cavity')

    p.add_argument('-fm', '--mech_f', type=lambda s: [float(f) for f in s.split(',')], required=True, help='list, frequencies of mech modes, Hz')
    p.add_argument('-qm', '--mech_q', type=lambda s: [float(q) for q in s.split(',')], required=True, help='list, qualify factors of mech modes')
    p.add_argument('-km', '--mech_k', type=lambda s: [float(k) for k in s.split(',')], required=True, help='list, K values, rad/s/(MV)^2')

    p.add_argument('-ib', '--beam_i', type=float, required=True, help='float, average beam current, A')
    p.add_argument('-nb', '--beam_n', type=int, required=True, help='int, number of time steps during beam, sample')

    p.add_argument('-nf', '--fill_n', type=int, required=True, help='int, number of time steps during cavity filling, sample')

    p.add_argument('-ts', '--step_t', type=float, required=True, help='float, time step duration, s')
    p.add_argument('-ns', '--step_n', type=int, required=True, help='int, number of time steps in total, sample')

    return p


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
        return self.obs_dw

    def step(self, action):

        # --! generate current microphonics
        dw_micr = 2.0 * np.pi * np.random.randn() * 0.2

        i = self.step_cnt

        status, self.state_cav, vr, self.obs_dw, self.state_mech = sim_scav_step(
            self.cav_wh,
            self.obs_dw,
            dw_micr,
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

        return self.obs_dw[0, 0], None, done

    def replay(self, ic, policy, obs_nsample=1, skip_nsample=0):
        pass

    @property
    def reward_fn(self):
        return self.r_fn

