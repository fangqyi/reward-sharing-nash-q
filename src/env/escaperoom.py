import numpy as np

from env.multiagentenv import MultiAgentEnv


class EscapeRoom(MultiAgentEnv):

    def __init__(self, args):

        self.args = args

        self.n_agents = self.args.n_agents
        self.name = 'er'
        self.l_action = 3
        # Observe self position (1-hot),
        # other agents' positions (1-hot for each other agent)
        # - total amount given to each other agent
        self.l_obs = 3 + 3*(self.n_agents - 1) # + (self.n_agents - 1)

        self.episode_limit = self.args.episode_limit
        self.min_at_lever = self.args.min_at_lever
        self.randomize = self.args.randomize

        self.actors = [Actor(idx, self.n_agents, self.l_obs)
                       for idx in range(self.n_agents)]

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, door_open):
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)

        if self.args.reward_sanity_check:
            rewards[0] = 10 if actions[0] == 1 else -1
            rewards[1] = 2 if actions[1] == 0 else -1
        else:
            for agent_id in range(0, self.n_agents):
                if door_open and actions[agent_id] == 2:
                    rewards[agent_id] = 10
                elif actions[agent_id] == self.actors[agent_id].position:
                    # no penalty for staying at current position
                    rewards[agent_id] = 0
                else:
                    rewards[agent_id] = -1

        return rewards

    def get_obs(self):
        list_obs = []
        for actor in self.actors:
            list_obs.append(actor.get_obs(self.state))

        return list_obs

    def step(self, actions): #, given_rewards):

        door_open = self.get_door_status(actions)
        rewards = self.calc_reward(actions, door_open)
        for idx, actor in enumerate(self.actors):
            actor.act(actions[idx]) #, given_rewards[idx])
        self.steps += 1
        self.state = [actor.position for actor in self.actors]
        list_obs_next = self.get_obs()

        # Terminate if (door is open and some agent ended up at door)
        # or reach max_steps
        done = (door_open and 2 in self.state) or self.steps == self.episode_limit

        return list_obs_next, rewards, done

    def reset(self):
        for actor in self.actors:
            actor.reset(self.randomize)
        self.state = [actor.position for actor in self.actors]
        self.steps = 0

    def get_obs_size(self):
        return self.l_obs

    def get_avail_actions(self):
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


class Actor(object):

    def __init__(self, agent_id, n_agents, l_obs):

        self.agent_id = agent_id
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.position = 1
        self.total_given = np.zeros(self.n_agents - 1)

    def act(self, action): #reward_given, observe_given=True):

        self.position = action
        # if observe_given:
        #    self.total_given += reward_given

    def get_obs(self, state): # observe_given=True):
        obs = np.zeros(self.l_obs)
        # position of self
        obs[state[self.agent_id]] = 1
        list_others = list(range(0, self.n_agents))
        del list_others[self.agent_id]
        # positions of other agents
        for idx, other_id in enumerate(list_others):
            obs[3*(idx + 1) + state[other_id]] = 1

        # total amount given to other agents
        # if observe_given:
        #     obs[-(self.n_agents - 1):] = self.total_given

        return obs

    def reset(self, randomize=False):
        if randomize:
            self.position = np.random.randint(3)
        else:
            self.position = 1
        self.total_given = np.zeros(self.n_agents - 1)
