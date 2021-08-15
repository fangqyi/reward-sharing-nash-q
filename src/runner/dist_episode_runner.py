"""
Code adapted from https://github.com/TonghanWang/ROMA
"""
import copy
import math

import numpy
import torch

from env import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class DistEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.n_agents = self.args.n_agents
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](self.args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.train_dis_returns = []
        self.test_returns = []
        self.test_dis_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, z_q, z_p, z_idx, test_mode=False, sample_return_mode=False, train_phase="pretrain"):  # run one eps
        self.reset()

        if isinstance(z_p, torch.Tensor):
            z_q = z_q.detach().cpu().numpy()
            z_p = z_p.detach().cpu().numpy()

        if len(z_q.shape) == 3:
            z_q_cp = z_q[0]
            z_p_cp = z_p[0]
        else:
            z_q_cp = z_q
            z_p_cp = z_p

        terminated = False
        episode_return = [0] * self.n_agents
        distributed_return = [0] * self.n_agents
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            if self.env.is_masked():
                pre_transition_data = {
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()],
                    "adjacent_agents": [self.env.get_adj()],
                    "z_p": z_p,
                    "z_q": z_q,
                    "z_idx": z_idx,
                }
            else:
                pre_transition_data = {
                    # "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()],
                    "adjacent_agents": [[[1] * self.n_agents] * self.n_agents],
                    "z_p": z_p,
                    "z_q": z_q,
                    "z_idx": z_idx,
                }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            rewards, terminated, env_info = self.env.step(actions[0])
            distributed_rewards = [0] * self.n_agents

            # calculate distance
            dist = []
            for giver in range(self.n_agents):
                dist.append(softmax([- distance(z_q_cp[giver], z_p_cp[receiver]) for receiver in range(self.n_agents)]))

            for receiver in range(self.n_agents):
                for giver in range(self.n_agents):
                    distributed_rewards[receiver] += dist[giver][receiver] * rewards[giver]

            episode_return = [episode_return[idx] + rewards[idx] for idx in range(self.n_agents)]
            distributed_return = [distributed_return[idx] + distributed_rewards[idx] for idx in range(self.n_agents)]

            post_transition_data = {
                "actions": actions,
                "rewards": rewards,
                "redistributed_rewards": distributed_rewards,
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        if self.env.is_masked():
            last_data = {
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "adjacent_agents": [self.env.get_adj()],
                "z_p": z_p,
                "z_q": z_q,
                "z_idx": z_idx,
            }
        else:
            last_data = {
                # "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "adjacent_agents": [[[1] * self.n_agents] * self.n_agents],
                "z_p": z_p,
                "z_q": z_q,
                "z_idx": z_idx,
            }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions, }, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_dis_returns = self.test_dis_returns if test_mode else self.train_dis_returns
        log_prefix = "test_" if test_mode else ""

        for k in set(cur_stats) | set(env_info):
            if isinstance(env_info.get(k), list):
                cur_stats.update({k: cur_stats.get(k, []) + env_info.get(k, 0)})
            else:
                cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        cur_dis_returns.append(distributed_return)
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_dis_returns, cur_stats, log_prefix, train_phase, z_q_cp, z_p_cp)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_dis_returns, cur_stats, log_prefix, train_phase, z_q_cp, z_p_cp)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if sample_return_mode:
            # return return (for training)
            return distributed_return
        else:
            # return sample batch
            return self.batch

    def _log(self, returns, dis_returns, stats, prefix, train_phase, z_q, z_p):
        # self.logger.console_logger.info("Agent rewards:")
        if train_phase == "pretrain" and prefix == "test_":
            struct_prefix = get_flat_str(z_q, "z_q:_") + "|" + get_flat_str(z_p, "z_p:_") + "|"
        else:
            struct_prefix = ""
        for agent_idx in range(self.n_agents):
            agent_returns = [dis_returns[t][agent_idx] for t in range(len(dis_returns))]
            self.logger.log_stat(
                "{}_{}{}agent {} post-sharing return_mean".format(train_phase, struct_prefix, prefix, agent_idx),
                np.mean(agent_returns),
                self.t_env)
            self.logger.log_stat(
                "{}_{}{}agent {} post-sharing return_std".format(train_phase, struct_prefix, prefix, agent_idx),
                np.std(agent_returns),
                self.t_env)
            agent_returns = [returns[t][agent_idx] for t in range(len(returns))]
            self.logger.log_stat(
                "{}_{}{}agent {} original return_mean".format(train_phase, struct_prefix, prefix, agent_idx),
                np.mean(agent_returns),
                self.t_env)
            self.logger.log_stat(
                "{}_{}{}agent {} original return_std".format(train_phase, struct_prefix, prefix, agent_idx),
                np.std(agent_returns),
                self.t_env)

        returns.clear()
        dis_returns.clear()

        for k, v in stats.items():
            if k != "n_episodes" and not isinstance(v, list):
                self.logger.log_stat("{} {}{}_mean".format(train_phase, prefix, k), v / stats["n_episodes"], self.t_env)
        stats.clear()


def get_flat_str(t, prefix):
    t = t.flatten()
    str_t = ""
    for i in range(t.shape[0]):
        str_t = "{}_{}".format(str_t, t[i]) if i != 0 else "{}".format(t[i])
    return prefix + str_t


def softmax(vector):
    e = [math.exp(x) for x in vector]
    return [x / sum(e) for x in e]


def distance(a, b):
    ret = numpy.linalg.norm(a - b, ord=2)
    return ret
