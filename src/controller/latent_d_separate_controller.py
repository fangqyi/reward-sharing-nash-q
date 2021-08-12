import torch as th
from torch.distributions import kl_divergence

from components.action_selectors import EpsilonGreedyActionSelector
from module.agents import REGISTRY as agent_REGISTRY

# This multi-agent controller shares parameters between agents
from module.utils import MLPMultiGaussianEncoder


class SeparateLatentMAC:
    def __init__(self, scheme, groups, args, train_phase):
        self.n_agents = args.n_agents
        self.train_phase = train_phase
        self.args = args
        self.scheme = scheme
        self._build_agents()

    def init_epsilon_schedule(self, phase):
        self.action_selector = EpsilonGreedyActionSelector(self.args, phase)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions.tolist()

    def sample_batch_latent_var(self, batch, t=None):
        # infer z that encodes the information about the current reward-sharing scheme
        # z_q, z_p: [n_agents][latent_relation_space_dim]
        vars = []
        for idx in range(self.n_agents):
            inputs = self._build_latent_encoder_input(batch, t)
            var = self.latent_encoders[idx].infer_posterior(inputs)
            if len(var.shape) == 1:
                var = var.unsqueeze(0)
            vars.append(var)
        return th.stack(vars, dim=1)

    def sample_latent_var(self, z_q, z_p):
        vars = []
        for idx in range(self.n_agents):
            inputs = th.cat([x.reshape(-1) for x in [z_q, z_p]], dim=-1).unsqueeze(0)
            vars.append(self.latent_encoders[idx].infer_posterior(inputs))
        return th.stack(vars, dim=1)

    def compute_kl_div(self):
        divs = []
        for idx in range(self.n_agents):
            divs.append(self.latent_encoders[idx].compute_kl_div())
            self.latent_encoders[idx].reset()
        divs = th.stack(divs, dim=1)
        return divs

    def compute_kl_div_agent(self, idx):
        enc_div = self.latent_encoders[idx].compute_kl_div() * self.args.encoder_kl_div_weight
        if self.args.mutual_information_reinforcement:
            enc_d = self.latent_encoders[idx].get_distribution()
            inf_d = self.inference_nets[idx].get_distribution()
            ce = enc_d.entropy().sum(dim=-1) * self.args.encoder_h_weight + kl_divergence(enc_d, inf_d).sum(dim=-1)*self.args.ce_kl_weight
            print(ce.shape)
            ce = th.clamp(ce, max=2e3)
            print(ce.shape)
            ce = th.log(1 + th.exp(ce)).unsqueeze(1)
            print(ce.shape)
            self.inference_nets[idx].reset()
            self.latent_encoders[idx].reset()
            return enc_div, ce
        self.latent_encoders[idx].reset()
        return enc_div

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if self.args.is_obs_image:
            obs_inputs, vec_inputs = agent_inputs
        avail_actions = ep_batch["avail_actions"][:, t]
        mask = ep_batch["adjacent_agents"][:, t]

        agent_outs = []
        for idx in range(self.n_agents):
            if self.args.is_obs_image:
                agent_input = [obs_inputs[idx], vec_inputs[idx]]
            else:
                agent_input = agent_inputs[idx]
            if self.args.agent == "dgn_agent":
                outputs = self.agents[idx](agent_input, mask, self.hidden_states[idx])
            else:
                outputs = self.agents[idx](agent_input, self.hidden_states[idx])
            if self.args.mutual_information_reinforcement:
                agent_out, self.hidden_states[idx], self.inference_inputs[idx] = outputs
            else:
                agent_out, self.hidden_states[idx] = outputs
            agent_outs.append(agent_out)
        agent_outs = th.stack(agent_outs, dim=1).reshape(ep_batch.batch_size * self.n_agents, -1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":  # (0, 1) -> (-inf, inf)
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def forward_agent(self, ep_batch, t, idx, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if self.args.is_obs_image:
            obs_inputs, vec_inputs = agent_inputs
        mask = ep_batch["adjacent_agents"][:, t, idx]

        if self.args.is_obs_image:
            agent_input = [obs_inputs[idx], vec_inputs[idx]]
        else:
            agent_input = agent_inputs[idx]
        if self.args.agent == "dgn_agent":
            agent_out, self.hidden_states[idx] = self.agents[idx](agent_input, mask, self.hidden_states[idx])
        else:
            agent_out, self.hidden_states[idx] = self.agents[idx](agent_input, self.hidden_states[idx])
        agent_out = agent_out.reshape(ep_batch.batch_size, -1)
        return agent_out

    def forward_inference_net_agent(self, idx):
        self.inference_nets[idx](self._build_inference_inputs_agent(idx))
        return self.inference_nets[idx].get_distribution()

    def init_hidden(self, batch_size):
        if 'init_hidden' in dir(self.agents[0]):
            self.hidden_states = [self.agents[idx].init_hidden().expand(batch_size, -1) for idx in range(self.n_agents)]
        if self.args.mutual_information_reinforcement:
            self.inference_inputs = [None] * self.n_agents

    def init_hidden_agent(self, batch_size, idx):
        if 'init_hidden' in dir(self.agents[0]):
            if len(self.hidden_states) != self.n_agents:
                self.hidden_states.append(self.agents[idx].init_hidden().expand(batch_size, -1))
            else:
                self.hidden_states[idx] = self.agents[idx].init_hidden().expand(batch_size, -1)
        if self.args.mutual_information_reinforcement:
            if len(self.inference_inputs) != self.n_agents:
                self.inference_inputs.append(None)
            else:
                self.inference_inputs[idx] = None

    def parameters(self):
        params = []
        for idx in range(self.n_agents):
            params.append(list(self.agents[idx].parameters()) + list(self.latent_encoders[idx].parameters()))
            if self.args.mutual_information_reinforcement:
                params.append(list(self.inference_nets[idx].parameters()))
        return params

    def load_state(self, other_mac):
        for idx in range(self.n_agents):
            self.agents[idx].load_state_dict(other_mac.agents[idx].state_dict())
            self.latent_encoders[idx].load_state_dict(other_mac.latent_encoders[idx].state_dict())
            if self.args.mutual_information_reinforcement:
                self.inference_nets[idx].load_state_dict(other_mac.inference_nets[idx].state_dict())

    def cuda(self):
        for idx in range(self.n_agents):
            self.agents[idx].cuda()
            self.latent_encoders[idx].cuda()
            if self.args.mutual_information_reinforcement:
                self.inference_nets[idx].cuda()

    def save_models(self, path):
        for idx in range(self.n_agents):
            th.save(self.agents[idx].state_dict(), "{}/agent{}.th".format(path, idx))
            th.save(self.latent_encoders[idx].state_dict(), "{}/encoder{}.th".format(path, idx))
            if self.args.mutual_information_reinforcement:
                th.save(self.inference_nets[idx].state_dict(), "{}/inference_net{}.th".format(path, idx))

    def load_models(self, path):
        for idx in range(self.n_agents):
            self.agents[idx].load_state_dict(
                th.load("{}/agent{}.th".format(path, idx), map_location=lambda storage, loc: storage))
            self.latent_encoders[idx].load_state_dict(
                th.load("{}/encoder{}.th".format(path, idx), map_location=lambda storage, loc: storage))
            if self.args.mutual_information_reinforcement:
                self.inference_nets[idx].load_state_dict(
                    th.load("{}/inference_net{}.th".format(path, idx), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.agent_output_type = self.args.agent_output_type
        self.agents = [agent_REGISTRY[self.args.agent](self.args, self.scheme) for _ in range(self.n_agents)]
        self.action_selector = EpsilonGreedyActionSelector(self.args, self.train_phase)

        latent_input_shape, latent_output_shape, latent_hidden_sizes = self._get_latent_shapes()
        self.latent_encoders = [MLPMultiGaussianEncoder(latent_input_shape, latent_output_shape, latent_hidden_sizes)
                                for _ in range(self.n_agents)]
        if self.args.mutual_information_reinforcement:
            self.inference_nets = [
                MLPMultiGaussianEncoder(self.agents[0].get_processed_output_shape(), latent_output_shape,
                                        latent_hidden_sizes) for _ in range(self.n_agents)]
        self.hidden_states = []
        self.inference_inputs = []

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        vec_inputs = []
        if self.args.obs_last_action:
            if t == 0:
                vec_inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                vec_inputs.append(batch["actions_onehot"][:, t - 1])
        vec_inputs.append(self.sample_batch_latent_var(batch, t))

        # process observation
        obs = batch["obs"][:, t]

        if self.args.is_obs_image:
            obs = obs.reshape(bs, self.n_agents, obs.shape[-3], obs.shape[-2],
                              obs.shape[-1])  # flatten the first two dims
            obs = list(th.split(obs, 1, dim=1))
            vec_inputs = th.cat(vec_inputs, dim=-1)
            vec_inputs = list(th.split(vec_inputs, 1, dim=1))
            for _ in range(self.n_agents):
                obs[_] = obs[_].squeeze(1)
                vec_inputs[_] = vec_inputs[_].squeeze(1)
            inputs = (obs, vec_inputs)  # return two objects as nn inputs
        else:
            inputs.append(obs)
            inputs.extend(vec_inputs)
            inputs = th.cat(inputs, dim=-1).split(1, dim=1)
            for _ in range(self.n_agents):
                inputs[_] = inputs[_].squeeze(1)  # return one, catting obs and other agent info together
        return inputs

    def _build_inference_inputs(self):
        return [th.cat(h.detach(), i.detach()) for h, i in zip(self.hidden_states, self.inference_inputs)]

    def _build_inference_inputs_agent(self, idx: int):
        return th.cat(self.hidden_states[idx], self.inference_inputs[idx])

    def _get_latent_shapes(self):
        return self.args.latent_relation_space_dim * self.n_agents * 2, \
               self.args.latent_var_dim, \
               self.args.latent_encoder_hidden_sizes

    def _build_latent_encoder_input(self, batch, t=None):
        # z_q, z_p: [n_agents][latent_relation_space_dim]
        bs = batch.batch_size
        if t is None:
            inputs = [batch["z_q"], batch["z_p"]]
        elif len(batch["z_q"][:, t].shape) == 1:  # FIXME: probably unnecessary
            inputs = [batch["z_q"][:, t].unsqueeze(0), batch["z_p"][:, t].unsqueeze(0)]
        else:
            inputs = [batch["z_q"][:, t], batch["z_p"][:, t]]

        return th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
