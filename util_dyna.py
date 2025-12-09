from abc import abstractmethod
from abc import ABC as interface

import numpy as np


class policy(interface):

    @abstractmethod
    def act(self, state):
        """Returns action for given ``state``."""
        return

    def __call__(self, state):
        return self.act(state)


class lqr(policy):

    def __init__(self, gain, noise=0.0):
        self.gain = gain
        self.noise = noise

    def act(self, state):
        u = -(self.gain @ state)
        return u + self.noise * np.random.standard_normal(size=u.shape)


class simulator(interface):

    @abstractmethod
    def simulate(self, ic, dt_control=1e-2, dt_sim=1e-4, t_final=100, skip_nsample=0):
        """Simulates a control system from an initial condition ``ic``."""
        return

