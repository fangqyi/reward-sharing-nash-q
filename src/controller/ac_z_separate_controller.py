import torch as th

from module.utils import MLPMultiGaussianEncoder
from module.utils.components import MultiSoftmaxMLP


class ZACSeparateMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.scheme = scheme
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
        inputs = th.cat([x.reshape(1, -1) for x in inputs], dim=-1)
        return inputs

    def _build_z_q_input(self, data, z_p):
        inputs = [data["z_p"], data["z_q"], z_p]
        inputs = th.cat([x.reshape(1, -1) for x in inputs], dim=-1)
        return inputs


class ZACDiscreteSeparateMAC(ZACSeparateMAC):
    def __init__(self, scheme, groups, args):
        super(ZACSeparateMAC, self).__init__(scheme, groups, args)
        self.relation_space_div_interval = args.relation_space_div_interval
        self.z_options = list(range(args.latent_relation_space_lower_bound,
                                    args.latent_relation_space_upper_bound,
                                    args.relation_space_div_interval))
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
        z_q_inputs = self._build_z_q_input(data, self.z_p_actors[idx].z)
        z_q_idx, prob_z_q = self.z_q_actors[idx].sample(z_q_inputs)
        if len(z_q_idx.shape) == 0:
            z_q = self.z_options[z_q_idx]
        else:
            z_q = [self.z_options[z_q_idx[idx]] for idx in range(len(z_q_idx))]
        return z_p, prob_z_p, z_q, prob_z_q 
