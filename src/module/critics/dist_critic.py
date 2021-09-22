import torch
import torch.nn.functional as F
from torch import nn

from module.utils.components import MLP
from utils.utils import identity, fanin_init


class CenDistCritic(nn.Module):  # Centralized critic that predicts the social welfare
    def __init__(self, scheme, args):
        super(CenDistCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape()
        output_shape = 1
        self.output_type = "q"

        self.critic = MLP(
            hidden_sizes=args.critic_hidden_sizes,
            input_size=input_shape,
            output_size=output_shape,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )

    def forward(self, batch):
        inputs = self._build_inputs(batch)
        return self.critic(inputs)

    def _build_inputs(self, batch):
        # assume latent_state: [bs, latent_state_size]
        # obs: [bs, seq_len, n_agents, obs_size]
        inputs = []

        z_p_s = batch["z_p"]  # [bs, n_agents, space_dim]
        z_q_s = batch["z_q"]  # [bs, n_agents, space_dim]
        inputs.append(z_q_s)
        inputs.append(z_p_s)

        # agent_id
        # agent_id = torch.eye(self.n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)
        # inputs.append(agent_id)

        inputs = torch.cat([x.reshape(-1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self):
        return self.n_agents * self.args.latent_relation_space_dim * 2  # z_q, z_q


class DecDistCritic(nn.Module):  # decentralized critic for individual rewards on a sharing scheme
    def __init__(self, scheme, args):
        super(DecDistCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape()
        output_shape = 1
        self.output_type = "q"

        self.critic = MLP(
            hidden_sizes=args.critic_hidden_sizes,
            input_size=input_shape,
            output_size=output_shape,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_params=None,
        )

    def forward(self, batch, latent_var=None, z_idx=None, *args):
        inputs = self._build_inputs(batch, latent_var, z_idx)
        return self.critic(inputs)

    def _build_inputs(self, batch, latent_var=None, z_idx=None):
        # assume latent_state: [bs, latent_state_size]
        # obs: [bs, seq_len, n_agents, obs_size]
        if len(batch["z_p"].shape) >= 2:
            bs = batch["z_p"].shape[0]
        else:
            bs = 1
        inputs = [batch["z_p"], batch["z_q"]]
        if self.args.sharing_scheme_encoder:
            inputs.append(latent_var)
        # used for critic updating agent's sharing scheme gradient (is it actually working?)
        if self.args.z_critic_gradient_update:
            inputs.append(z_idx)

        inputs = torch.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        print("inputs shape {}".format(inputs.shape))
        return inputs

    def _get_input_shape(self):
        shape = self.args.latent_relation_space_dim * 2 * self.n_agents
        if self.args.sharing_scheme_encoder:  # need to add encoded content
            shape += self.args.latent_var_dim  # encoded z_q, z_q
        if self.args.z_critic_gradient_update:
            shape += self.args.latent_relation_space_dim * 2
        return shape
