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
            self.conv = nn.Conv2d(in_channels=c, out_channels=args.conv_out_dim, kernel_size=args.kernel_size, stride=args.stride)
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
        #h = F.relu(self.fc3(h))
        q = self.fc2(h)
        return q, h

    def _get_conv_output_shape(self):  # ignore padding
        h = (self.args.obs_height-self.args.kernel_size)/self.args.stride + 1
        w = (self.args.obs_width-self.args.kernel_size)/self.args.stride + 1
        return h*w*self.args.conv_out_dim


class RNNAgentImageVec(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(RNNAgentImageVec, self).__init__()
        image_input_shape, vec_input_shape = input_shape
        self.args = args
        c, h, w = image_input_shape
        # if h != args.obs_height or w != args.obs_weight:
        #     print("input shape not matched with specified obs height or weight in args")
        #     raise ValueError
        self.conv = nn.Conv2d(in_channels=c, out_channels=args.conv_out_dim, kernel_size=args.kernel_size, stride=args.stride)
        input_dim = self._get_conv_output_shape() + vec_input_shape
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, output_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        image_inputs, vec_inputs = inputs
        x = nn.Flatten(F.relu(self.conv(image_inputs)))
        x = torch.cat([x, vec_inputs], axis=-1)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        #h = F.relu(self.fc3(h))
        q = self.fc2(h)
        return q, h

    def _get_conv_output_shape(self):  # ignore padding
        h = (self.args.obs_height-self.args.kernel_size)/self.args.stride + 1
        w = (self.args.obs_width-self.args.kernel_size)/self.args.stride + 1
        return h*w*self.args.conv_out_dim