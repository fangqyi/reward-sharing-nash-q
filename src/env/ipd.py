import numpy as np
from env.multiagentenv import MultiAgentEnv


class IteratedPrisonerDilemma(MultiAgentEnv):
    """
    A two-agent vectorized environment for the Prisoner's Dilemma game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    Adapted from https://raw.githubusercontent.com/alshedivat/lola/master/lola/envs/prisoners_dilemma.py
    """
    NAME = 'IPD'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, args):
        self.args = args
        if args.n_agent != self.NUM_AGENTS:
            print("Warning: Iterated prisoner dilemma currently only supports 2-agent environment, arg.n_agent = {}".format(args.n_agent))
            raise ValueError
        self.episode_limit = args.episode_limit
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])
        self.action_space = self.NUM_ACTIONS
        self.observation_space = self.NUM_STATES
        self._init_obs()
        self.step_count = None

    def _init_obs(self):
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        self.observations = [init_state, init_state]

    def reset(self):
        self.step_count = 0
        self._init_obs()

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]

        state = np.zeros(self.NUM_STATES)
        state[ac0 * 2 + ac1] = 1
        self.observations = [state, state]

        done = (self.step_count == self.episode_limit)

        return self.observations, rewards, done

    def get_obs(self):
        return self.observations

    def get_obs_size(self):
        return self.observation_space

    def get_state(self):
        return self.observations

    def get_state_size(self):
        return self.observation_space

    def get_avail_actions(self):
        return [[0, 1] for _ in range(self.NUM_AGENTS)]

    def get_total_actions(self):
        return self.NUM_ACTIONS

    def is_masked(self):
        return False

    def close(self):
        pass

    def get_env_info(self):
        env_info = {
                    "obs_shape": self.get_obs_size(),
                    "reward_shape": self.args.n_agents,
                    "n_actions": self.get_total_actions(),
                    "adjacent_agents_shape": self.NUM_AGENTS,
                    "n_agents": self.args.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info