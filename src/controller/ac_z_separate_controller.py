import torch as th

from components.action_selectors import EpsilonGreedyActionSelector
from module.utils import MLPMultiGaussianEncoder
from module.utils.components import MultiSoftmaxMLP, MultiMLP


class ZACSeparateMAC:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.args = args
        self.z_p_actors = [MLPMultiGaussianEncoder(input_size=args.latent_relation_space_dim * args.n_agents * 2,
                                                   output_size=args.latent_relation_space_dim,
                                                   mlp_hidden_sizes=args.latent_encoder_hidden_sizes,
                                                   sample_clamped=True,
                                                   clamp_lower_bound=args.latent_relation_space_lower_bound,
                                                   clamp_upper_bound=args.latent_relation_space_upper_bound)
                           for _ in range(self.args.n_agents)]
        self.z_q_actors = [MLPMultiGaussianEncoder(input_size=(args.latent_relation_space_dim * args.n_agents * 2
                                                               + args.latent_relation_space_dim),
                                                   output_size=args.latent_relation_space_dim,
                                                   mlp_hidden_sizes=args.latent_encoder_hidden_sizes,
                                                   sample_clamped=True,
                                                   clamp_lower_bound=args.latent_relation_space_lower_bound,
                                                   clamp_upper_bound=args.latent_relation_space_upper_bound)
                           for _ in range(self.args.n_agents)]

    def forward(self, data):
        z_p, prob_z_p, z_q, prob_z_q = [], [], [], []
        for i in range(self.n_agents):
            z_p_i, prob_z_p_i, z_q_i, prob_z_q_i = self.forward_agent(data, i)
            z_p.append(z_p_i)
            z_q.append(z_q_i)
            prob_z_p.append(prob_z_p_i)
            prob_z_q.append(prob_z_q_i)
        return th.stack(z_p, dim=0).view(1, self.args.n_agents, self.args.latent_relation_space_dim), \
               th.stack(prob_z_p, dim=0), \
               th.stack(z_q, dim=0).view(1, self.args.n_agents, self.args.latent_relation_space_dim), \
               th.stack(prob_z_q, dim=0)

    def forward_agent(self, data, idx):
        z_p_inputs = self._build_z_p_input(data)
        self.z_p_actors[idx](z_p_inputs)
        z_q_inputs = self._build_z_q_input(data, self.z_p_actors[idx].z)
        self.z_q_actors[idx](z_q_inputs)
        return self.z_p_actors[idx].z.detach(), self.z_p_actors[idx].prob_z, self.z_q_actors[idx].z.detach(), \
               self.z_q_actors[idx].prob_z

    def parameters(self):
        return [list(self.z_p_actors[i].parameters()) + list(self.z_q_actors[i].parameters()) for i in
                range(self.n_agents)]

    def load_state(self, other_mac):
        for idx in range(self.n_agents):
            self.z_p_actors[idx].load_state_dict(other_mac.latent_encoders[idx].state_dict())
            self.z_q_actors[idx].load_state_dict(other_mac.latent_encoders[idx].state_dict())

    def cuda(self):
        for idx in range(self.n_agents):
            self.z_p_actors[idx].cuda()
            self.z_q_actors[idx].cuda()

    def save_models(self, path):
        for idx in range(self.n_agents):
            th.save(self.z_p_actors[idx].state_dict(), "{}/z_p_actor{}.th".format(path, idx))
            th.save(self.z_q_actors[idx].state_dict(), "{}/z_q_actor{}.th".format(path, idx))

    def load_models(self, path):
        for idx in range(self.n_agents):
            self.z_p_actors[idx].load_state_dict(
                th.load("{}/z_p_actor{}.th".format(path, idx), map_location=lambda storage, loc: storage))
            self.z_q_actors[idx].load_state_dict(
                th.load("{}/z_q_actor{}.th".format(path, idx), map_location=lambda storage, loc: storage))

    def _build_z_p_input(self, data):
        inputs = [data["z_p"], data["z_q"]]
        for x in inputs:   # shotgun fixes
            if x.device != self.args.device:
                x.to(self.args.device)
        print("z_p_input device")
        inputs = th.cat([x.reshape(1, -1) for x in inputs], dim=-1)
        print(inputs.device)
        return inputs

    def _build_z_q_input(self, data, z_p):
        inputs = [data["z_p"], data["z_q"], z_p]
        for x in inputs:
            if x.device != self.args.device:
                x.to(self.args.device)
        print("z_q_input device")
        inputs = th.cat([x.reshape(1, -1) for x in inputs], dim=-1)
        print(inputs.device)
        return inputs


class ZACDiscreteSeparateMAC(ZACSeparateMAC):
    def __init__(self, args):
        super(ZACDiscreteSeparateMAC, self).__init__(args)
        self.relation_space_div_interval = args.relation_space_div_interval
        self.z_options = [args.latent_relation_space_lower_bound + idx * args.relation_space_div_interval
                          for idx in range(int(args.latent_relation_space_lower_bound),
                                           int(args.latent_relation_space_upper_bound),
                                           int(args.relation_space_div_interval))]  # this is horrible

        output_size = args.latent_relation_space_dim * len(self.z_options)
        self.z_p_actors = [MultiSoftmaxMLP(input_size=args.latent_relation_space_dim * args.n_agents * 2,
                                           output_size=output_size,
                                           hidden_sizes=args.latent_encoder_hidden_sizes,
                                           head_num=len(self.z_options))
                           for _ in range(self.args.n_agents)]
        self.z_q_actors = [MultiSoftmaxMLP(input_size=(args.latent_relation_space_dim * args.n_agents * 2
                                                       + args.latent_relation_space_dim),
                                           output_size=output_size,
                                           hidden_sizes=args.latent_encoder_hidden_sizes,
                                           head_num=len(self.z_options))
                           for _ in range(self.args.n_agents)]

    def forward_agent(self, data, idx):
        z_p_inputs = self._build_z_p_input(data)
        z_p_idx, prob_z_p = self.z_p_actors[idx].sample(z_p_inputs)
        if len(z_p_idx.shape) == 0:
            z_p = self.z_options[z_p_idx]
        else:
            z_p = [self.z_options[z_p_idx[idx]] for idx in range(len(z_p_idx))]
        z_p = th.tensor(z_p).to(self.args.device)
        z_q_inputs = self._build_z_q_input(data, z_p)
        z_q_idx, prob_z_q = self.z_q_actors[idx].sample(z_q_inputs)
        if len(z_q_idx.shape) == 0:
            z_q = self.z_options[z_q_idx]
        else:
            z_q = [self.z_options[z_q_idx[idx]] for idx in range(len(z_q_idx))]
        z_q = th.tensor(z_q).to(self.args.device)
        return z_p, prob_z_p, z_q, prob_z_q


class ZQSeparateMAC(ZACSeparateMAC):
    def __init__(self, args):
        super(ZQSeparateMAC, self).__init__(args)
        self.relation_space_div_interval = args.relation_space_div_interval
        self.z_options = [args.latent_relation_space_lower_bound + idx * args.relation_space_div_interval
                          for idx in range(int(args.latent_relation_space_lower_bound),
                                           int(args.latent_relation_space_upper_bound),
                                           int(args.relation_space_div_interval))]  # this is horrible

        output_size = args.latent_relation_space_dim * len(self.z_options)
        self.z_p_actors = [MultiMLP(input_size=args.latent_relation_space_dim * args.n_agents * 2,
                                    output_size=output_size,
                                    hidden_sizes=args.latent_encoder_hidden_sizes,
                                    head_num=len(self.z_options))
                           for _ in range(self.args.n_agents)]
        self.z_q_actors = [MultiMLP(input_size=(args.latent_relation_space_dim * args.n_agents * 2
                                                + args.latent_relation_space_dim),
                                    output_size=output_size,
                                    hidden_sizes=args.latent_encoder_hidden_sizes,
                                    head_num=len(self.z_options))
                           for _ in range(self.args.n_agents)]
        self.z_p_actors_selector = EpsilonGreedyActionSelector(self.args, "z_train")
        self.z_q_actors_selector = EpsilonGreedyActionSelector(self.args, "z_train")

    def forward_agent(self, data, idx):  # should only be used in training
        z_p_vals = self.forward_z_p(data, idx)
        # print("cur_z_p shape{}".format(data["cur_z_p"].shape))
        z_q_vals = self.forward_z_q(data, idx, data["cur_z_p"].view(self.n_agents, -1)[idx])
        return z_p_vals, z_q_vals

    def forward(self, data):  # should only be used in training
        z_p_vals, z_q_vals = [], []
        for i in range(self.n_agents):
            z_p_val, z_q_val = self.forward_agent(data, i)
            z_p_vals.append(z_p_val)
            z_q_vals.append(z_q_val)
        return th.stack(z_p_vals, dim=0).view(self.args.n_agents, self.args.latent_relation_space_dim, -1), \
               th.stack(z_q_vals, dim=0).view(self.args.n_agents, self.args.latent_relation_space_dim, -1)

    def forward_z_p(self, data, idx):
        z_p_inputs = self._build_z_p_input(data)
        # print("z_p_inputs shape{}".format(z_p_inputs.shape))
        z_p_q_vals = self.z_p_actors[idx].forward(z_p_inputs)  # [z_dim, div_num]
        return z_p_q_vals

    def forward_z_q(self, data, idx, z_p):
        z_q_inputs = self._build_z_q_input(data, z_p)
        # print("z_q_inputs shape{}".format(z_q_inputs.shape))
        z_q_q_vals = self.z_q_actors[idx].forward(z_q_inputs)
        return z_q_q_vals

    def select_z(self, data, t_env, test_mode=False):
        chosen_z_p_s, chosen_z_q_s = [], []
        chosen_z_p_idx_s, chosen_z_q_idx_s = [], []
        for idx in range(self.n_agents):
            # z_p
            z_p_outs = self.forward_z_p(data, idx).view(1, self.args.latent_relation_space_dim, -1)
            chosen_z_p_idx = self.z_p_actors_selector.select_action(z_p_outs, t_env=t_env, test_mode=test_mode).view(-1)
            z_p = th.tensor([self.z_options[chosen_z_p_idx[idx]] for idx in range(len(chosen_z_p_idx))]).float().to(self.args.device)
            chosen_z_p_s.append(z_p)
            chosen_z_p_idx_s.append(chosen_z_p_idx)
            # z_q
            z_q_outs = self.forward_z_q(data, idx, z_p.clone()).view(1, self.args.latent_relation_space_dim,
                                                                                -1)
            chosen_z_q_idx = self.z_q_actors_selector.select_action(z_q_outs, t_env=t_env, test_mode=test_mode).view(-1)
            z_q = th.tensor([self.z_options[chosen_z_q_idx[idx]] for idx in range(len(chosen_z_q_idx))]).float()
            chosen_z_q_s.append(z_q)
            chosen_z_q_idx_s.append(chosen_z_q_idx)
        # stack
        chosen_z_p_s = th.stack(chosen_z_p_s, dim=0)  # [n_agents, space_dim]
        chosen_z_q_s = th.stack(chosen_z_q_s, dim=0)
        chosen_z_p_idx_s = th.stack(chosen_z_p_idx_s, dim=0)
        chosen_z_q_idx_s = th.stack(chosen_z_q_idx_s, dim=0)
        return chosen_z_p_s, chosen_z_q_s, chosen_z_p_idx_s, chosen_z_q_idx_s
