import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        if args.is_obs_image:
            c, h, w = input_shape
            if h != args.obs_height or w != args.obs_weight:
                print("input shape not matched with specified obs height or weight in args")
                raise ValueError
            self.conv = nn.Conv2d(in_channels=c, out_channels=args.conv_out_dim, kernel_size=args.kernel_size,
                                  stride=args.stride)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(self._get_conv_output_shape(), args.rnn_hidden_dim)
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if self.args.is_obs_image:
            x = self.flatten(F.relu(self.conv(inputs)))
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # h = F.relu(self.fc3(h))
        q = self.fc2(h)
        return q, h

    def _get_conv_output_shape(self):  # ignore padding
        h = (self.args.obs_height - self.args.kernel_size) / self.args.stride + 1
        w = (self.args.obs_width - self.args.kernel_size) / self.args.stride + 1
        return int(h * w * self.args.conv_out_dim)


class RNNAgentImageVec(nn.Module):
    def __init__(self, args, scheme):
        super(RNNAgentImageVec, self).__init__()
        self.args = args
        self.scheme = scheme
        c, h, w = scheme["obs"]["vshape"]
        self.conv = nn.Conv2d(in_channels=c, out_channels=args.conv_out_dim, kernel_size=args.kernel_size, stride=args.stride)
        self.flatten = nn.Flatten()
        input_shape = self._get_vec_input_shape(scheme) + self._get_conv_output_shape()
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        image_inputs, vec_inputs = inputs
        x = self.flatten(F.relu(self.conv(image_inputs)))
        x = torch.cat([x, vec_inputs], axis=-1)
        y = self.fc1(x)
        y = F.relu(y)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(y, h_in)
        # h = F.relu(self.fc3(h))
        q = self.fc2(h)
        if self.args.mutual_information_reinforcement:
            return q, h, x
        else:
            return q, h

    def get_processed_output_shape(self):
        return self._get_vec_input_shape(self.scheme) + self._get_conv_output_shape() + self.args.rnn_hidden_dim

    def _get_conv_output_shape(self):  # ignore padding
        h = (self.args.obs_height - self.args.kernel_size) / self.args.stride + 1
        w = (self.args.obs_width - self.args.kernel_size) / self.args.stride + 1
        return int(h * w * self.args.conv_out_dim)

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