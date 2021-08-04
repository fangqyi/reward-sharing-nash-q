import copy

import torch

from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam

from module.critics.dist_critic import CentralizedDistCritic, DecentralizedDistCritic


class MetaQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = self.args.n_agents
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        if args.centralized_social_welfare:
            # centralized social welfare critic: all the agents optimize the estimated social utility
            self.z_critic = CentralizedDistCritic(scheme, args)
            self.z_critic_params = list(self.z_critic.parameters())
            self.z_critic_optimiser = Adam(params=self.z_critic_params, lr=args.z_critic_lr,
                                           eps=args.optim_eps)
        else:
            # fully decentralized critic on reward-sharing structure
            self.z_critics = [DecentralizedDistCritic(scheme, args) for _ in range(self.args.n_agents)]
            self.z_critic_params = sum([list(critic.parameters()) for critic in self.z_critics], [])
            self.z_critic_optimiser = Adam(params=self.z_critic_params, lr=args.z_critic_lr, eps=args.optim_eps)

        if args.separate_agents:
            self.optimisers = [Adam(params=self.params[idx], lr=args.lr, eps=args.optim_eps) for idx in range(self.n_agents)]
        else:
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)

        # betas=(args.optim_beta1, args.optim_beta2)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def get_social_welfare_z(self, entry):
        # returns the estimated social utility
        if self.args.centralized_social_welfare:
            z_vals = self.z_critic(entry)
        else:
            latent_vars = self.mac.sample_latent_var(entry["z_q"], entry["z_p"])
            z_vals = sum([self.z_critics[i](entry, i, latent_vars) for i in range(self.args.n_agents)])
        return z_vals

    def z_train(self, entry, t_env):
        # train centralized critic
        if self.args.centralized_social_welfare:
            z_vals = self.z_critic(entry)
            z_critic_loss = ((z_vals - entry["evals"]) ** 2).sum()
        else:
            latent_vars = self.mac.sample_latent_var(entry["z_q"], entry["z_p"])
            z_vals = [self.z_critics[i](entry, i, latent_vars) for i in range(self.args.n_agents)]
            z_critic_loss = sum([(z_vals[i] - entry["evals"][i])**2 for i in range(self.args.n_agents)])

        self.logger.log_stat("z_critic_loss", z_critic_loss.item(), t_env)
        self.z_critic_optimiser.zero_grad()
        z_critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.z_critic_params, self.args.grad_norm_clip)  # max_norm
        self.z_critic_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("z_critic_loss", z_critic_loss.item(), t_env)
            self.logger.log_stat("z_critic_grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if self.args.separate_agents:
            self._separate_train(batch, t_env, episode_num)
        else:
            self._train(batch, t_env, episode_num)

    def _train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["redistributed_rewards"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # self.mac.init_latent(batch.batch_size)

        kl_divs = []
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # (bs,n,n_actions)
            kl_divs.append(self.mac.compute_kl_div())
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        kl_divs = th.stack(kl_divs, dim=1)[:, :-1]
        # (bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        # self.target_mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)  # (bs,n,n_actions)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values

        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net
        # (bs,t,1)

        kl_mask = copy.deepcopy(mask).expand_as(kl_divs)
        masked_kl_div = kl_divs * kl_mask
        kl_div_loss = masked_kl_div.sum() / kl_mask.sum()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data (LSE)
        td_error_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = td_error_loss + kl_div_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("policy_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("kl_div_abs", kl_div_loss.abs().item(), t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _separate_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["redistributed_rewards"][:, :-1]  # [bs, t, n_agents, -1]
        actions = batch["actions"][:, :-1]  # [bs, t, n_agents, -1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]  # [bs, t, n_agents, -1]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # self.mac.init_latent(batch.batch_size)

        kl_divs = []
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # (bs,n,n_actions)
            kl_div = self.mac.compute_kl_div()
            kl_divs.append(kl_div)  # (bs,n, ))
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        kl_divs = th.stack(kl_divs, dim=1)[:, :-1]
        # (bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        # self.target_mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)  # (bs,n,n_actions)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values

        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net
        # (bs,t,1)

        kl_divs = list(kl_divs.split(1, dim=2))
        td_errors = list(td_error.split(1, dim=2))
        for idx in range(self.n_agents):
            print("agent {}".format(idx))
            kl_divs[idx] = kl_divs[idx].squeeze(2)
            kl_mask = copy.deepcopy(mask).expand_as(kl_divs[idx])
            masked_kl_div = kl_divs[idx] # * kl_mask
            kl_div_loss = masked_kl_div.sum() # / kl_mask.sum()

            td_mask = copy.deepcopy(mask).expand_as(td_errors[idx])
            # 0-out the targets that came from padded data
            masked_td_error = td_errors[idx] # * td_mask
            # Normal L2 loss, take mean over actual data (LSE)
            td_error_loss = (masked_td_error ** 2).sum() # / td_mask.sum()

            loss = td_error_loss + kl_div_loss

            # Optimise
            self.optimisers[idx].zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params[idx], self.args.grad_norm_clip)  # max_norm
            self.optimisers[idx].step()

            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                self.logger.log_stat("loss_{}".format(idx), loss.item(), t_env)
                self.logger.log_stat("policy_grad_norm_{}".format(idx), grad_norm, t_env)
                mask_elems = mask.sum().item()
                self.logger.log_stat("kl_div_abs_{}".format(idx), kl_div_loss.abs().item(), t_env)
                self.logger.log_stat("td_error_abs_{}".format(idx), (masked_td_error.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("q_taken_mean_{}".format(idx),
                                     (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                                     t_env)
                self.logger.log_stat("target_mean_{}".format(idx), (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                     t_env)
                self.log_stats_t = t_env

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.args.centralized_social_welfare:
            self.z_critic.cuda()
        else:
            for z_critic in self.z_critics:
                z_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.args.separate_agents:
            for idx in range(self.n_agents):
                th.save(self.optimisers[idx].state_dict(), "{}/opt{}.th".format(path, idx))
        else:
            th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.args.separate_agents:
            for idx in range(self.n_agents):
                self.optimisers[idx].load_state_dict(
                    th.load("{}/opt{}.th".format(path, idx), map_location=lambda storage, loc: storage))
        else:
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
