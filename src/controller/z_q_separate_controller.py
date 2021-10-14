import torch as th

from components.action_selectors import EpsilonGreedyActionSelector
from module.utils.components import MultiMLP


# Q Learning Controller
# Fixed a point (for agents only to learn distance)
# Alternative game (filter out own strategy from last rounds)

class ZQSeparateMAC():
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.args = args
        self.relation_space_div_interval = args.relation_space_div_interval
        self.z_options = [0, 3, 5]
        # list(range(int(args.latent_relation_space_lower_bound), int(args.latent_relation_space_upper_bound) + 1, int(args.relation_space_div_interval)))  # this is horrible

        input_size = args.latent_relation_space_dim * (args.n_agents - 1) * 2
        output_size = args.latent_relation_space_dim * len(self.z_options)  # TODO: is actor functional when dim > 1

        # agent 0, z_q fixed to 0
        if args.latent_relation_space_dim >= 1:
            self.z_q_actors = [MultiMLP(input_size=input_size + args.latent_relation_space_dim,
                                        output_size=output_size - len(self.z_options),
                                        hidden_sizes=args.latent_encoder_hidden_sizes,
                                        head_num=len(self.z_options))]
        else:
            self.z_q_actors = [None]
        self.z_q_actors.extend([MultiMLP(input_size=input_size + args.latent_relation_space_dim,
                                         output_size=output_size,
                                         hidden_sizes=args.latent_encoder_hidden_sizes,
                                         head_num=len(self.z_options))
                                for _ in range(0, self.args.n_agents - 1)])

        self.z_p_actors = [MultiMLP(input_size=input_size,
                                    output_size=output_size,
                                    hidden_sizes=args.latent_encoder_hidden_sizes,
                                    head_num=len(self.z_options))
                           for _ in range(self.args.n_agents)]
        # print("input size: {}".format(input_size + args.latent_relation_space_dim))
        self.z_p_actors_selector = EpsilonGreedyActionSelector(self.args, "z_train")
        self.z_q_actors_selector = EpsilonGreedyActionSelector(self.args, "z_train")

    def forward_agent(self, data, idx, is_train):  # should only be used in training
        if is_train:
            cur_z_p = data["cur_z_p"]
        else:
            cur_z_p = None
        z_p_vals = self._forward_z_p(data, idx)
        if idx != 0:
            z_q_vals = self._forward_z_q(data, idx, cur_z_p, is_train)
        else:
            z_q_vals = th.zeros(z_p_vals.shape)
        return z_p_vals, z_q_vals

    def select_z(self, data, t_env, test_mode=False):
        chosen_z_p_s, chosen_z_q_s = [], []
        chosen_z_p_idx_s, chosen_z_q_idx_s = [], []
        for idx in range(self.n_agents):
            z_p, z_q, chosen_z_p_idx, chosen_z_q_idx = self.select_z_agent(data, t_env, idx, test_mode)
            # z_p
            chosen_z_p_s.append(z_p)
            chosen_z_p_idx_s.append(chosen_z_p_idx)
            # z_q
            chosen_z_q_s.append(z_q)
            chosen_z_q_idx_s.append(chosen_z_q_idx)
        # stack
        chosen_z_p_s = th.stack(chosen_z_p_s, dim=0)  # [n_agents, space_dim]
        chosen_z_q_s = th.stack(chosen_z_q_s, dim=0)
        chosen_z_p_idx_s = th.stack(chosen_z_p_idx_s, dim=0)
        chosen_z_q_idx_s = th.stack(chosen_z_q_idx_s, dim=0)
        return chosen_z_p_s, chosen_z_q_s, chosen_z_p_idx_s, chosen_z_q_idx_s

    def select_z_agent(self, data, t_env, idx, test_mode=False):

        # z_p
        z_p_outs = self._forward_z_p(data, idx).view(1, self.args.latent_relation_space_dim, -1)
        chosen_z_p_idx = self.z_p_actors_selector.select_action(z_p_outs, t_env=t_env, test_mode=test_mode).view(-1).to(
            self.args.device)
        z_p = th.tensor([self.z_options[chosen_z_p_idx[idx]] for idx in range(len(chosen_z_p_idx))]).float()

        # z_q
        if idx != 0:
            z_q_outs = self._forward_z_q(data, idx, z_p, is_train=False).view(1, self.args.latent_relation_space_dim,
                                                                              -1)
            chosen_z_q_idx = self.z_p_actors_selector.select_action(z_q_outs, t_env=t_env, test_mode=test_mode).view(
                -1).to(self.args.device)
            z_q = th.tensor([self.z_options[chosen_z_q_idx[idx]] for idx in range(len(chosen_z_q_idx))]).float().to(
                self.args.device)
        else:  # fixed the value for last agent
            chosen_z_q_idx = th.zeros([self.args.latent_relation_space_dim]).to(self.args.device)
            z_q = th.zeros([self.args.latent_relation_space_dim]).float().to(self.args.device)

        return z_p, z_q, chosen_z_p_idx, chosen_z_q_idx

    def parameters(self):
        return [list(self.z_p_actors[i].parameters()) + list(self.z_q_actors[i].parameters()) if i != 0 else [] for i in
                range(self.n_agents)]

    def load_state(self, other_mac):
        for idx in range(self.n_agents):
            self.z_p_actors[idx].load_state_dict(other_mac.z_p_actors[idx].state_dict())
            if idx != 0:
                self.z_q_actors[idx].load_state_dict(other_mac.z_q_actors[idx].state_dict())

    def cuda(self):
        for idx in range(self.n_agents):
            self.z_p_actors[idx].cuda()
            if idx != 0:
                self.z_q_actors[idx].cuda()

    def save_models(self, path):
        for idx in range(self.n_agents):
            th.save(self.z_p_actors[idx].state_dict(), "{}/z_p_actor{}.th".format(path, idx))
            if idx != 0:
                th.save(self.z_q_actors[idx].state_dict(), "{}/z_q_actor{}.th".format(path, idx))

    def load_models(self, path):
        for idx in range(self.n_agents):
            if idx == 0 and self.args.latent_relation_space_dim == 1:
                self.z_q_actors[idx].load_state_dict(
                    th.load("{}/z_q_actor{}.th".format(path, idx), map_location=lambda storage, loc: storage))
            else:
                self.z_p_actors[idx].load_state_dict(
                    th.load("{}/z_p_actor{}.th".format(path, idx), map_location=lambda storage, loc: storage))
                self.z_q_actors[idx].load_state_dict(
                    th.load("{}/z_q_actor{}.th".format(path, idx), map_location=lambda storage, loc: storage))

    def _build_z_p_input(self, data, idx):
        inputs = [data["z_p"], data["z_q"]]
        if len(data["z_p"].shape) >= 2:
            bs = data["z_p"].shape[0]
        else:
            bs = 1
        filtered_inputs = []
        for x in inputs:
            x = x.reshape(bs, self.n_agents, -1)
            filtered_inputs.append(x[:, th.arange(self.n_agents) != idx])
        inputs = th.cat([x.reshape(bs, -1) for x in filtered_inputs], dim=-1)
        return inputs

    def _build_z_q_input(self, data, z_p, idx, is_train=False):
        if len(data["z_p"].shape) >= 2:
            bs = data["z_p"].shape[0]
        else:
            bs = 1
        inputs = [data["z_p"], data["z_q"]]
        filtered_inputs = []
        for x in inputs:
            x = x.reshape(bs, self.n_agents, -1)
            filtered_inputs.append(x[:, th.arange(self.n_agents) != idx])
        if is_train:
            z_p = z_p.view(bs, self.n_agents, -1)[:, idx]
        filtered_inputs.append(z_p)
        inputs = th.cat([x.reshape(bs, -1) for x in filtered_inputs], dim=-1)
        return inputs

    def _forward_z_p(self, data, idx):
        z_p_inputs = self._build_z_p_input(data, idx)
        z_p_q_vals = self.z_p_actors[idx].forward(z_p_inputs)  # [z_dim, div_num]
        return z_p_q_vals

    def _forward_z_q(self, data, idx, z_p, is_train=False):
        z_q_inputs = self._build_z_q_input(data, z_p, idx, is_train)
        z_q_q_vals = self.z_q_actors[idx].forward(z_q_inputs)
        return z_q_q_vals
