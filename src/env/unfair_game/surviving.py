import numpy as np
import random
import copy

from env.multiagentenv import MultiAgentEnv


def is_legal(x, y):
    return (x >= 1) & (x <= 30) & (y >= 1) & (y <= 30)


class Surviving(MultiAgentEnv):
    def __init__(self, args):
        super(Surviving, self).__init__()
        self.args = args
        self.redistributed = random.shuffle(list(range(args.n_agents)))
        self.episode_limit = args.episode_limit
        self.n_agent = args.n_agents
        self.n_action = 5
        self.max_food = args.max_food
        self.capability = 2 * self.n_agent

        self.maze = self.build_env()
        self.ants = []
        for i in range(self.n_agent):
            self.ants.append([np.random.randint(0, 30) + 1, np.random.randint(0, 30) + 1])

        self.foods = []
        for i in range(self.n_agent):
            self.foods.append(self.max_food)

        self.n_resource = args.n_resource
        self.resource = []
        self.resource_pos = []
        for i in range(self.n_resource):
            self.resource_pos.append([np.random.randint(0, 30) + 1, np.random.randint(0, 30) + 1])
            self.resource.append(np.random.randint(100, 120))

        self.steps = 0
        self.len_obs = 29

    def reset(self):

        self.maze = self.build_env()

        self.ants = []
        for i in range(self.n_agent):
            self.ants.append([np.random.randint(0, 30) + 1, np.random.randint(0, 30) + 1])

        self.foods = []
        for i in range(self.n_agent):
            self.foods.append(self.max_food)

        self.resource = []
        self.resource_pos = []
        for i in range(self.n_resource):
            self.resource_pos.append([np.random.randint(0, 30) + 1, np.random.randint(0, 30) + 1])
            self.resource.append(np.random.randint(100, 120))

    def build_env(self):

        maze = np.zeros((32, 32))
        for i in range(32):
            maze[0][i] = -1
            maze[i][0] = -1
            maze[31][i] = -1
            maze[i][31] = -1

        return maze

    def get_obs(self):

        obs = []

        maze_ant = np.zeros((32, 32))
        for index in range(self.n_agent):
            x = self.ants[index][0]
            y = self.ants[index][1]
            maze_ant[x][y] = 1

        for index in range(self.n_agent):
            h = []
            x = self.ants[index][0]
            y = self.ants[index][1]
            for i in range(5):
                h.append(np.mod(x, 2))
                x = int(x / 2)
            for i in range(5):
                h.append(np.mod(y, 2))
                y = int(y / 2)
            x_t = self.ants[index][0]
            y_t = self.ants[index][1]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    h.append(self.maze[x_t + i][y_t + j])

            for i in range(-1, 2):
                for j in range(-1, 2):
                    h.append(maze_ant[x_t + i][y_t + j])

            h.append(self.foods[index])
            obs.append(h)

        return obs

    def get_obs_size(self):
        return self.len_obs

    def get_total_actions(self):
        return self.n_action

    def get_avail_actions(self):
        return [[0, 1, 2, 3, 4] for _ in range(self.n_agent)]

    def close(self):
        pass

    def get_adj(self):

        adj = []

        maze_ant = np.ones((32, 32), dtype=np.int) * -1
        for index in range(self.n_agent):
            x = self.ants[index][0]
            y = self.ants[index][1]
            maze_ant[x][y] = index

        for index in range(self.n_agent):
            agent_adj = np.zeros(self.n_agent)
            x = self.ants[index][0]
            y = self.ants[index][1]

            for i in range(-3, 4):
                for j in range(-3, 4):
                    if is_legal(x + i, y + j):
                        if maze_ant[x + i][y + j] != -1:
                            agent_adj[maze_ant[x + i][y + j]] = 1
            adj.append(agent_adj)

        return adj

    def step(self, actions):
        env_info = []
        for i in range(self.n_agent):
            x = self.ants[i][0]
            y = self.ants[i][1]

            if actions[i] == 0:
                if self.maze[x - 1][y] != -1:
                    self.ants[i][0] = x - 1
            if actions[i] == 1:
                if self.maze[x + 1][y] != -1:
                    self.ants[i][0] = x + 1
            if actions[i] == 2:
                if self.maze[x][y - 1] != -1:
                    self.ants[i][1] = y - 1
            if actions[i] == 3:
                if self.maze[x][y + 1] != -1:
                    self.ants[i][1] = y + 1
            if actions[i] == 4:
                self.foods[i] += 2 * self.maze[x][y]
                self.maze[x][y] = 0

            self.foods[i] = max(0, min(self.foods[i] - 1, self.max_food))

        reward = [0.4] * self.n_agent
        for i in range(self.n_agent):
            if self.foods[i] == 0:
                reward[i] = - 0.2

        redistributed_reward = [0] * self.n_agent
        print(self.redistributed)
        for i in range(self.n_agent):
            print(redistributed_reward[i])
            print(self.redistributed[i])
            print(reward[self.redistributed[i]])
            redistributed_reward[i] = reward[self.redistributed[i]]

        done = False

        if (self.maze.sum() + 120) > self.capability:
            return reward, done

        for i in range(self.n_resource):

            x = self.resource_pos[i][0] + np.random.randint(-3, 4)
            y = self.resource_pos[i][1] + np.random.randint(-3, 4)

            if is_legal(x, y):

                num = np.random.randint(1, 6)
                self.maze[x][y] += num
                self.maze[x][y] = min(self.maze[x][y], 5)
                self.resource[i] -= num

                if self.resource[i] <= 0:
                    self.resource_pos[i][0] = np.random.randint(0, 30) + 1
                    self.resource_pos[i][1] = np.random.randint(0, 30) + 1
                    self.resource[i] = np.random.randint(100, 120)

        return reward, done, env_info

    def is_masked(self):
        return True

    def get_env_info(self):
        env_info = {
                    "obs_shape": self.get_obs_size(),
                    "reward_shape": self.args.n_agents,
                    "n_actions": self.get_total_actions(),
                    "adjacent_agents_shape": self.args.n_agents,
                    "n_agents": self.args.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
