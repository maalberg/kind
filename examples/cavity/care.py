# --! care: cavity resonance environment --!


from abc import abstractmethod
from abc import ABC as interface

import argparse
import numpy as np

import care_control
import care_sim


def create_args_parser():
    """Creates parser for care arguments."""

    p = argparse.ArgumentParser(description='CARE: cavity resonance environment')

    p.add_argument('-fg', '--gen_f', type=float, required=True, help='float, carrier frequency, Hz')
    p.add_argument('-ag', '--gen_a', type=float, required=False, default=1, help='float, amplitude of generator, V')
    p.add_argument('-pg', '--gen_p', type=float, required=False, default=0, help='float, phase of generator, rad')
    p.add_argument('-ig', '--gen_i', type=float, required=True, help='float, forward power equivalent current, A')
    p.add_argument('-og', '--gen_o', type=float, required=True, help='float, offset frequency from carrier, Hz')

    p.add_argument('-ga', '--amp_g', type=float, required=True, help='float, amplifier gain, unitless')

    p.add_argument('-rc', '--cav_r', type=float, required=True, help='float, R over Q of cavity, Ohm')
    p.add_argument('-qc', '--cav_q', type=float, required=True, help='float, loaded quality factor of cavity, unitless')

    p.add_argument('-fm', '--mech_f', type=lambda s: [float(f) for f in s.split(',')], required=True, help='list, frequencies of mech modes, Hz')
    p.add_argument('-qm', '--mech_q', type=lambda s: [float(q) for q in s.split(',')], required=True, help='list, qualify factors of mech modes')
    p.add_argument('-km', '--mech_k', type=lambda s: [float(k) for k in s.split(',')], required=True, help='list, K values, rad/s/(MV)^2')

    p.add_argument('-ib', '--beam_i', type=float, required=True, help='float, average beam current, A')
    p.add_argument('-nb', '--beam_n', type=int, required=True, help='int, number of time steps during beam, sample')

    p.add_argument('-nf', '--fill_n', type=int, required=True, help='int, number of time steps during cavity filling, sample')

    p.add_argument('-ts', '--step_t', type=float, required=True, help='float, time step duration, s')
    p.add_argument('-ns', '--step_n', type=int, required=True, help='int, number of time steps in total, sample')

    return p


class care(interface):

    @abstractmethod
    def reset(self):
        """Resets this cavity resonance entity."""
        pass

    @abstractmethod
    def step(self, *args):
        """
        Advances the state of this cavity resonance entity by one time step.

        Parameters:
            args: tuple, concrete parameters depend on a particular entity

        Returns:
            result: concrete result depends on a particular entity
        """
        pass


class simulator(care):

    def __init__(self, args):

        self.src = source(args)
        self.amp = amplifier(args)
        self.cav = cavity(args)

    def reset(self):
        self.src.reset()
        self.amp.reset()
        self.cav.reset()

    def step(self, action):

        vg = self.src.step()
        vf = self.amp.step(vg)
        vb = 0
        vc, dc = self.cav.step(vf, vb, action)

        return vc, dc


class timing(care):

    def __init__(self, rf_fill_nstep, rf_flat_nstep):

        self.rf_fill_nstep = rf_fill_nstep
        self.rf_flat_nstep = rf_flat_nstep
        self.cur_step = 0

        # --! list of observers that will be notified when events happen
        self.observers = []

    def reset(self):
        self.cur_step = 0

    def step(self, *args):

        if self.cur_step==0:
            self.notify_rf_pulse(started=True)
        elif self.cur_step==self.rf_fill_nstep:
            pass
        elif self.cur_step==(self.rf_fill_nstep + self.rf_flat_nstep):
            self.notify_rf_pulse(started=False)

        self.cur_step += 1

    def register_observer(self, observer):
        self.observers.append(observer)

    def notify_rf_pulse(self, started=True):

        for observer in self.observers:
            if started:
                observer.rf_pulse_started()
            else:
                observer.rf_pulse_finished()


class care_observer(interface):

    @abstractmethod
    def rf_pulse_started(self):
        pass

    @abstractmethod
    def rf_pulse_finished(self):
        pass

    @abstractmethod
    def beam_pulse_started(self):
        pass


class source(care, care_observer):

    def __init__(self, args):

        # --! save source configuration
        self.a = args.gen_a    # amplitude
        self.p = args.gen_p    # phase
        self.f = args.gen_o    # frequency
        self.t = args.step_t   # time step

        # --! current phase that acts as the state of this source
        self.cur_p = self.p

    def reset(self):
        self.cur_p = self.p

    def step(self, *args):

        # --! advance current phase
        self.cur_p = self.cur_p + 2.0 * np.pi * self.f * self.t

        # --! generate a complex sinusoid rotating at frequency f relative to the simulator reference frame
        return self.a * np.exp(1j*self.cur_p)

    def rf_pulse_started(self):
        pass

    def rf_pulse_finished(self):
        pass

    def beam_pulse_started(self):
        pass


class amplifier(care):

    def __init__(self, args):
        self.gain_db = 20*np.log10(args.amp_g)

    def reset(self):
        pass

    def step(self, *args):
        vg, = args
        return vg * 10.0**(self.gain_db / 20.0)


class cavity(care):

    def __init__(self, args):

        # --! create electrical and mechanical cavity matrices
        self.me = care_control.create_matrices_elec(args.gen_f, args.cav_q)
        self.mm = care_control.create_matrices_mech(np.array(args.mech_f), np.array(args.mech_q), np.array(args.mech_k))

        # --! initialize cavity voltage and states to zero initial condition
        self.vc = 0.0
        self.sm = np.zeros(self.mm.b.shape)

        self.dt = args.step_t

    def reset(self):

        # --! reset states
        self.vc = 0.0
        self.sm = np.zeros(self.mm.b.shape)

    def step(self, *args):
        """
        Advances cavity state by one time step.

        Parameters:
            args: tuple:
                vf: complex, forward voltage, V
                vb: complex, beam voltage, V
                a:  float, tuner action, rad/s

        Returns:
            vc: complex, cavity voltage, V
            dc: float, cavity detuning, rad/s
        """

        # --! unpack forward and beam voltages, as well as action
        vf, vb, a = args

        # --! advance cavity state by one step
        self.vc, dc, self.sm = care_sim.step_cav(self.vc, vf, self.sm, self.me, self.mm, a=a, dt=self.dt, vb=vb)

        # --! return cavity voltage and detuning
        return self.vc, dc

