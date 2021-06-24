"""
Wrapper around Sequential Social Dilemma environment.
Code adapted from https://github.com/011235813/lio/blob/master/lio/env/ssd.py
"""
import numpy as np
from lio.env import maps
from social_dilemmas.envs.cleanup import CleanupEnv

from env.multiagentenv import MultiAgentEnv
from utils.configdict import ConfigDict


class Cleanup(MultiAgentEnv):

    def __init__(self, args):

        self.name = 'ssd'
        self.args = args
        self.dim_obs = [3, self.args.obs_height,
                        self.args.obs_width]
        self.episode_limit = self.args.episode_limit

        self.cleaning_penalty = self.args.cleaning_penalty
        # Original space (not necessarily in this order, see
        # the original ssd files):
        # no-op, up, down, left, right, turn-ccw, turn-cw, penalty, clean
        if (self.args.disable_left_right_action and
                self.args.disable_rotation_action):
            self.l_action = 4
            self.cleaning_action_idx = 3
            # up, down, no-op, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 8}
        elif self.args.disable_left_right_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # up, down, no-op, rotate cw, rotate ccw, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
        elif self.args.disable_rotation_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # left, right, up, down, no-op, clean
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8}
        else:  # full action space except penalty beam
            self.l_action = 8
            self.cleaning_action_idx = 7
            # Don't allow penalty beam
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8}

        self.obs_cleaned_1hot = self.args.obs_cleaned_1hot

        self.n_agents = self.args.n_agents

        if self.args.map_name == 'cleanup_small_sym':
            ascii_map = maps.CLEANUP_SMALL_SYM
        elif self.args.map_name == 'cleanup_10x10_sym':
            ascii_map = maps.CLEANUP_10x10_SYM

        cleanup_params = ConfigDict()
        cleanup_params.appleRespawnProbability = args.appleRespawnProbability
        cleanup_params.thresholdDepletion = args.thresholdDepletion
        cleanup_params.thresholdRestoration = args.thresholdRestoration
        cleanup_params.wasteSpawnProbability = args.wasteSpawnProbability

        self.env = CleanupEnv(ascii_map=ascii_map,
                              num_agents=self.n_agents, render=False,
                              shuffle_spawn=self.args.shuffle_spawn,
                              global_ref_point=self.args.global_ref_point,
                              view_size=self.args.view_size,
                              random_orientation=self.args.random_orientation,
                              cleanup_params=cleanup_params,
                              beam_width=self.args.beam_width)

        # length of action input to learned reward function
        if self.args.obs_cleaned_1hot:
            self.l_action_for_r = 2
        else:
            self.l_action_for_r = self.l_action

        self.obs = None
        self.steps = 0

    def process_obs(self, obs_dict):  # adjusted the dims for convnet
        processed_obs = [obs / 256.0 for obs in list(obs_dict.values())]
        processed_obs = np.moveaxis(np.array(processed_obs), -1, 1)
        return processed_obs

    def reset(self):
        """Resets the environemnt.

        Returns:
            List of agent observations
        """
        self.obs = self.env.reset()
        self.steps = 0

    def step(self, actions):
        """Takes a step in env.

        Args:
            actions: list of integers

        Returns:
            List of observations, list of rewards, done, info
        """
        actions = [self.map_to_orig[a] for a in actions]
        actions_dict = {'agent-%d' % idx: actions[idx]
                        for idx in range(self.n_agents)}

        # all objects returned by env.step are dicts
        obs_next, rewards, dones, info = self.env.step(actions_dict)
        self.steps += 1

        obs_next = self.process_obs(obs_next)
        rewards = list(rewards.values())
        if self.cleaning_penalty > 0:
            for idx in range(self.n_agents):
                if actions[idx] == 8:
                    rewards[idx] -= self.cleaning_penalty

        # done = dones['__all__']  # apparently they hardcode done to False
        done = dones['__all__'] or self.steps == self.episode_limit

        return obs_next, rewards, done, info

    def render(self):
        self.env.render()

    def get_obs(self):
        return self.process_obs(self.obs)

    def get_obs_size(self):
        return self.dim_obs

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        raise NotImplementedError

    def get_avail_actions(self):  # unfiltered
        return [[_ for _ in range(self.l_action)] for _ in range(self.n_agents)]

    def get_total_actions(self):
        return self.l_action

    def is_masked(self):
        return False

    def close(self):
        pass

    def get_env_info(self):
        env_info = {"obs_shape": self.get_obs_size(),
                    "reward_shape": self.n_agents,
                    "n_actions": self.get_total_actions(),
                    "adjacent_agents_shape": self.n_agents,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
