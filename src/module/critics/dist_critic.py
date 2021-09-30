import math

import torch
import torch.nn.functional as F
from torch import nn

from module.utils.components import MLP
from utils.utils import identity, fanin_init


class DistCritic(nn.Module):  # decentralized critic for individual rewards on a sharing scheme
    def __init__(self, scheme, args):
        super(DistCritic, self).__init__()

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

        if self.args.is_alt_upd_z:
            inputs = [batch["cur_z_p"], batch["cur_z_q"]]
        else:
            inputs = [batch["z_p"], batch["z_q"]]

        if self.args.sharing_scheme_encoder:
            inputs.append(latent_var)
        # used for critic updating agent's sharing scheme gradient (is it actually working?)
        if self.args.z_critic_gradient_update:
            inputs.append(z_idx)

        inputs = torch.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        # print("inputs shape {}".format(inputs.shape))
        return inputs

    def _get_input_shape(self):
        shape = self.args.latent_relation_space_dim * 2 * self.n_agents
        if self.args.sharing_scheme_encoder:  # need to add encoded content
            shape += self.args.latent_var_dim  # encoded z_q, z_q
        if self.args.z_critic_gradient_update:
            shape += self.args.latent_relation_space_dim * 2
        return shape

# class UniDistCritic(nn.Module):  # decentralized

class MultiDistCritic(nn.Module):  # decentralized critic for individual rewards on all sharing schemes
    def __init__(self, scheme, args):
        super(MultiDistCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        l = int(self.args.latent_reladtion_space_lower_bound)
        u = int(self.args.latent_relation_space_upper_bound)
        intrvl = int(self.args.relation_space_div_interval)
        self.num_options = len(list(range(l, u+1, intrvl)))

        input_shape = self._get_input_shape()
        output_shape = self._get_output_shape()
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

        if self.args.is_alt_upd_z:
            inputs = [batch["cur_z_p"], batch["cur_z_q"]]
        else:
            inputs = [batch["z_p"], batch["z_q"]]

        if self.args.sharing_scheme_encoder:
            inputs.append(latent_var)
        # used for critic updating agent's sharing scheme gradient (is it actually working?)
        if self.args.z_critic_gradient_update:
            inputs.append(z_idx)

        inputs = torch.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        # print("inputs shape {}".format(inputs.shape))
        return inputs

    def _get_input_shape(self):
        shape = self.args.latent_relation_space_dim * 2 * self.n_agents
        if self.args.sharing_scheme_encoder:  # need to add encoded content
            shape += self.args.latent_var_dim  # encoded z_q, z_q
        if self.args.z_critic_gradient_update:
            shape += self.args.latent_relation_space_dim * 2
        return shape

    def _get_output_shape(self):
        return math.pow(self.num_options, self.args.latent_relation_space_dim)**2
