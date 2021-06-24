import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))
        return out


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class DGNAgent(nn.Module):
    def __init__(self, args, scheme):
        super(DGNAgent, self).__init__()
        self.args = args
        if args.is_obs_image:
            c, h, w = scheme["obs"]["vshape"]
            # if h != args.obs_height or w != args.obs_weight:
            #     print("input shape not matched with specified obs height or weight in args")
            #     raise ValueError
            self.conv = nn.Conv2d(in_channels=c, out_channels=args.conv_out_dim, kernel_size=args.kernel_size, stride=args.stride)
            input_shape = self._get_vec_input_shape(scheme) + args.conv_out_dim
        else:
            input_shape = self._get_input_shape(scheme)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.encoder = Encoder(input_shape, args.hidden_dim)
        self.att_1 = AttModel(args.n_agents, args.hidden_dim, args.hidden_dim, args.hidden_dim)
        self.att_2 = AttModel(args.n_agents, args.hidden_dim, args.hidden_dim, args.hidden_dim)
        self.q_net = Q_Net(args.hidden_dim, args.n_actions)

    def forward(self, x, mask, _):
        image_x, vec_x = x
        if self.args.is_obs_image:
            x = F.relu(self.conv(image_x))
            print(x.shape)
            print(vec_x.shape)
            x = torch.cat([x, vec_x], axis=-1)
        x = x.view(mask.shape[0], self.args.n_agents, -1)
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask)
        q = self.q_net(h3)
        return q, []

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"] + self._get_vec_input_shape(scheme)
        return input_shape

    def _get_vec_input_shape(self, scheme):
        input_shape = 0
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.args.n_agents
        if self.args.meta_type == "pq":
            input_shape += self.args.n_agents*self.args.n_agents*2
        if self.args.meta_type == "distance_latent":
            input_shape += self.args.latent_var_dim
        return input_shape

