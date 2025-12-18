from abc import abstractmethod
from abc import ABC as interface

from collections import deque
from collections import namedtuple

import torch
import numpy as np

state_preview = namedtuple('state_preview', 'state preview')


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


class environment(interface):

    @abstractmethod
    def reset(self):
        """Resets this environment to start a new episode and returns the first observation from that new episode."""
        return

    @abstractmethod
    def step(self, u):
        """Applies action ``u`` and returns the next observation, reward and termination flags."""
        return


class kind_rl_environment(environment):
    def __init__(self, environment_true, model, dataset, policy_base, reward):

        self.env_true = environment_true
        self.model = model
        self.dataset = dataset
        self.policy_base = policy_base
        self.reward = reward

        self.lookback_x = deque(maxlen=model.args.lookback_nsample)
        self.lookback_u = deque(maxlen=model.args.lookback_nsample)

    def reset(self):
        """Resets this environment to start a new episode and returns the first observation from the new episode."""

        self.lookback_x.clear()
        self.lookback_u.clear()

        # --! reset a true environment and obtain its initial condition
        x = self.env_true.reset()

        # --! fill lookback windows with real states and actions
        for step in range(lookback_nsample):

            u = self.policy_base(x)

            self.lookback_x.append(x)
            self.lookback_u.append(u)

            # --! next state
            x = self.env_true.step(u)

        # --! save the last next state and return it - this becomes the current observation for an RL agent
        #
        # --! note that the state lookback window is ahead of action window by one
        self.lookback_x.append(x)
        return x
 
    def step(self, u):
        """Applies action ``u`` and returns the next observation, reward and termination flags."""

        # --! take the current observation that RL reacted to by issuing action u
        x = self.lookback_x[-1]

        # --! action u from RL is treated as residual control, which together with a baseline policy forms the final action u
        #
        # --! the final action is saved in its lookback window, thus making the state and action windows equal in size
        u = self.policy_base(x) + u
        self.lookback_u.append(u)

        # --! as we have a state and action pair we can immediately query current reward
        r = self.reward(x, u)

        # --! moreover, as we have balanced lookback windows we can call kind model to get the next state forecast
        forecast = self.model(self.dataset, self._get_lookback_tensor())

        # --! act on the true environment with a new RL-based action to get the real next state
        #
        # --! note that if the forecast above is from +1 to +H, then the real next state is +1
        x = self.env_true.step(u)

        # --! finally, test a done condition
        done = self.env_true.jstep >= self.env_true.nstep - self.model.args.forecast_nsample

        # --! return the next state with preview, reward and a done flag
        return state_preview(x, forecast), r, done

    def _get_lookback_tensor(self):
        """Returns the lookback windows converted from deques to a single tensor."""
        lookback_x = torch.tensor(self.lookback_x)
        lookback_u = torch.tensor(self.lookback_u).unsqueeze(-1)
        mask_u = torch.ones_like(lookback_u)

        return torch.unsqueeze(torch.cat([lookback_x, lookback_u, mask_u], dim=-1), 0)

