import copy

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch


class MetaLearner:
    def __init__(self, mac, scheme, logger, args, z_mac=None):
        self.args = args
        self.n_agents = self.args.n_agents
        self.mac = mac
        if z_mac is not None:
            self.z_mac = z_mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        # optimiser for z actor if needed
        self.z_actors_params = self.z_mac.parameters()
        self.z_actors_optimisers = [Adam(params=self.z_actors_params[i], lr=args.z_actor_lr, eps=args.optim_eps) for
                                    i
                                    in range(self.n_agents)]

        # agent optimiser(s)
        self.optimisers = [Adam(params=self.params[idx], lr=args.lr, eps=args.optim_eps) for idx in
                           range(self.n_agents)]

        self.target_mac = copy.deepcopy(mac)
        # not needed if agent(s) has no regards for future rounds
        # if args.z_q_update:
        #     self.target_z_mac = copy.deepcopy(z_mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def z_train(self, entry, t_env):  # train z critic
        bs = entry["z_q"].shape[0]
        is_logging = False
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.log_stats_t = t_env

        for i in range(self.n_agents):
            z_p_vals, z_q_vals = self.z_mac.forward_agent(entry, i, is_train="True")
            # TODO: Double check the gather
            chosen_z_q_vals = th.gather(z_q_vals, dim=-1, index=entry["cur_z_q_idx"].view(bs, self.n_agents, -1)[:, i]).squeeze(-1)
            # print("evals shape {}".format(entry["evals"].shape))
            # fake td_error
            if i == self.n_agents - 1 and self.args.latent_relation_space_dim == 1:
                loss = (chosen_z_q_vals - entry["evals"].view(bs, self.n_agents)[:, i].detach()).sum()
            else:
                chosen_z_p_vals = th.gather(z_p_vals, dim=-1,
                                            index=entry["cur_z_p_idx"].view(bs, self.n_agents, -1)[:, i]).squeeze(-1)
                loss = (chosen_z_p_vals + chosen_z_q_vals - entry["evals"].view(bs, self.n_agents)[:, i].detach()).sum()
            self.z_actors_optimisers[i].zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.z_actors_params[i], self.args.grad_norm_clip)
            self.z_actors_optimisers[i].step()

            if is_logging:
                self.logger.log_stat("z_actors_loss_agent_{}".format(i), loss.item(), t_env)
                self.logger.log_stat("z_actors_grad_norm_{}".format(i), grad_norm, t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):  # train agent models
        # train agents
        if self.args.separate_agents:
            for idx in range(self.n_agents):
                self._train_agent(batch, t_env, idx)
        else:
            self._train(batch, t_env)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _get_latent_vars(self, entry):
        return self.mac.sample_latent_var(entry["z_q"], entry["z_p"]) if self.args.sharing_scheme_encoder else None

    def _train(self, batch: EpisodeBatch, t_env: int):
        # TODO: implement mutual information
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

        kl_divs = 0
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # (bs,n,n_actions)
            kl_divs += self.mac.compute_kl_div()
            mac_out.append(agent_outs)  # [t,(bs,n,n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        kl_divs /= batch.max_seq_length
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

        kl_div_loss = kl_divs

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

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("policy_grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("kl_div_abs", kl_div_loss.abs().item(), t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _train_agent(self, batch: EpisodeBatch, t_env: int, idx: int):
        # Get the relevant quantities
        rewards = batch["redistributed_rewards"][:, :-1, idx]  # [bs, t, n_agents, -1]
        actions = batch["actions"][:, :-1, idx]  # [bs, t, n_agents, -1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :, idx]  # [bs, t, n_agents, -1]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden_agent(batch.batch_size, idx)
        # self.mac.init_latent(batch.batch_size)

        kl_divs = 0
        ce_losses = 0
        for t in range(batch.max_seq_length):
            agent_out = self.mac.forward_agent(batch, idx=idx, t=t)  # (bs,n_actions)
            mac_out.append(agent_out)  # [t,(bs,n_actions)]
            if self.args.mutual_information_reinforcement and self.args.sharing_scheme_encoder:
                self.mac.forward_inference_net_agent(idx=idx)
                kl_div, ce_loss = self.mac.compute_kl_div_agent(idx=idx)
                ce_losses += ce_loss
                kl_divs += kl_div  # (bs, ))
            elif not self.args.mutual_information_reinforcement and self.args.sharing_scheme_encoder:
                kl_div = self.mac.compute_kl_div_agent(idx=idx)
                kl_divs += kl_div
            if not self.args.sharing_scheme_encoder and self.args.mutual_information_reinforcement:
                z_idx = batch["z_idx"][:, t]
                ce = self.mac.compute_kl_div_agent(idx=idx, z_idx=z_idx)
                ce_losses += ce
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        kl_divs /= batch.max_seq_length
        ce_losses /= batch.max_seq_length
        # (bs,t,n,n_actions), Q values of n_actions

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=2, index=actions).squeeze(-1)  # Remove the last dim
        # (bs,t,n) Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden_agent(batch.batch_size, idx)  # (bs,hidden_size)
        # self.target_mac.init_latent(batch.batch_size)

        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward_agent(batch, idx=idx, t=t)  # (bs,n_actions)
            target_mac_out.append(target_agent_outs)  # [t,(bs,n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index
        # (bs,t,n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values

        target_max_qvals = target_mac_out.max(dim=2)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated).squeeze(-1) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach()).unsqueeze(-1)  # no gradient through target net
        # (bs,t,1)

        td_mask = copy.deepcopy(mask).expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * td_mask
        # Normal L2 loss, take mean over actual data (LSE)
        td_error_loss = (masked_td_error ** 2).sum() / td_mask.sum()
        loss = td_error_loss

        if self.args.sharing_scheme_encoder:
            loss += kl_divs
        if self.args.mutual_information_reinforcement:
            loss += ce_losses

        # Optimise
        self.optimisers[idx].zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params[idx], self.args.grad_norm_clip)  # max_norm
        self.optimisers[idx].step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("agent{}_loss".format(idx), loss.item(), t_env)
            self.logger.log_stat("agent{}_policy_grad_norm".format(idx), grad_norm, t_env)
            mask_elems = td_mask.sum().item()
            if self.args.sharing_scheme_encoder:
                self.logger.log_stat("agent{}_kl_div_abs".format(idx), kl_divs.abs().item(), t_env)
            if self.args.mutual_information_reinforcement:
                self.logger.log_stat("agent{}_ce_loss_abs".format(idx), ce_losses.abs().item(), t_env)
            self.logger.log_stat("agent{}_td_error_abs".format(idx), (masked_td_error.abs().sum().item() / mask_elems),
                                 t_env)
            self.logger.log_stat("agent{}_target_mean".format(idx),
                                 (targets.unsqueeze(-1) * td_mask).sum().item() / (mask_elems),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.z_mac.cuda()

    def save_models(self, path, train=False):
        self.mac.save_models(path)
        for idx in range(self.n_agents):
            th.save(self.optimisers[idx].state_dict(), "{}/opt{}.th".format(path, idx))

        if train:
            self.z_mac.save_models(path)

    def load_models(self, path, train=False):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        for idx in range(self.n_agents):
            self.optimisers[idx].load_state_dict(
                th.load("{}/opt{}.th".format(path, idx), map_location=lambda storage, loc: storage))

        if train:  # also load z critic(s) and respective optimisers
            for idx in range(self.n_agents):
                self.z_critic.load_state_dict(
                    th.load("{}/z_critic{}.th".format(path, idx), map_location=lambda storage, loc: storage))
                self.z_critic_optimiser.load_state_dict(
                    th.load("{}/z_critic_opt{}.th".format(path, idx), map_location=lambda storage, loc: storage))

        self.z_mac.load_models(path)
