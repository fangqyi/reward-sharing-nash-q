import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as D
from torch.distributions import kl_divergence

from utils.utils import identity, fanin_init, product_of_gaussians, LayerNorm


class MLPMultiGaussianEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 mlp_hidden_sizes,
                 sample_clamped=False,
                 clamp_upper_bound=None,
                 clamp_lower_bound=None,
                 mlp_init_w=3e-3,
                 mlp_hidden_activation=F.leaky_relu,
                 mlp_output_activation=identity,
                 mlp_hidden_init=fanin_init,
                 mlp_bias_init_value=0.1,
                 mlp_layer_norm=False,
                 mlp_layer_norm_params=None,
                 use_information_bottleneck=True,
                 ):
        super(MLPMultiGaussianEncoder, self).__init__()
        self.mlp = FlattenMLP(
            input_size=input_size,
            output_size=2 * output_size if use_information_bottleneck else output_size,  # vars + means
            hidden_sizes=mlp_hidden_sizes,
            init_w=mlp_init_w,
            hidden_activation=mlp_hidden_activation,
            output_activation=mlp_output_activation,
            hidden_init=mlp_hidden_init,
            b_init_value=mlp_bias_init_value,
            layer_norm=mlp_layer_norm,
            layer_norm_params=mlp_layer_norm_params,
        )
        self.use_information_bottleneck = use_information_bottleneck
        self.input_size = input_size
        self.output_size = output_size
        self.sample_clamped = sample_clamped
        self.clamp_upper_bound = clamp_upper_bound
        self.clamp_lower_bound = clamp_lower_bound
        self.z_means = None
        self.z_vars = None
        self.z = None
        self.log_prob_z = None
        self.prob_z = None

    def infer_posterior(self, inputs):
        self.forward(inputs)
        return self.z

    def sample_z(self):
        if self.use_information_bottleneck:
            posteriors = D.Normal(self.z_means, torch.sqrt(self.z_vars))
            self.z = posteriors.rsample()
            # print("sample got {} with mean {} and vars {}".format(self.z.item(), self.z_means.item(),  self.z_vars.item()))
            self.log_prob_z = posteriors.log_prob(self.z)
            self.prob_z = torch.exp(self.log_prob_z)
            # print("probability got {}".format(self.prob_z.item()))
        else:
            self.z = self.z_means
        if self.sample_clamped:
            self.z[:] = self.z.clone().clamp(self.clamp_lower_bound,
                                             self.clamp_upper_bound)  # TODO: double check if it cancels gradient at border

    def forward(self, input):
        # print("input")
        # print(input)
        params = self.mlp(input)  # [batch_size, 2*output_size]
        if self.use_information_bottleneck:
            self.z_means = params[..., :self.output_size]
            # print("raw z_vars")
            # print(params[..., self.output_size:])
            self.z_vars = F.softplus(params[..., self.output_size:])
            # z_params = [product_of_gaussians(m, s) for m,s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        else:
            self.z_means = torch.mean(params, dim=1)  # FIXME: doublecheck
            self.z_vars = None
        self.sample_z()

    def get_distribution(self):
        return D.Normal(self.z_means, torch.sqrt(self.z_vars))

    def compute_kl_div(self):
        device = self.z_means[0].device
        prior = D.Normal(torch.zeros(self.output_size).to(device),
                         torch.ones(self.output_size).to(device))
        post = D.Normal(self.z_means, torch.sqrt(self.z_vars))
        kl_divs = kl_divergence(post, prior).sum(dim=-1).mean()
        return kl_divs

    def reset(self):
        self.z_means = None
        self.z_vars = None


class MLP(nn.Module):
    # https://github.com/katerakelly/oyster/blob/master/rlkit/torch/networks.py
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes,
                 init_w=3e-3,
                 hidden_activation=F.leaky_relu,
                 output_activation=identity,
                 hidden_init=fanin_init,
                 b_init_value=0.1,
                 layer_norm=False,
                 layer_norm_params=None,
                 ):
        super(MLP, self).__init__()
        if layer_norm_params is None:
            layer_norm_params = dict()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm

        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivation=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivation:
            return output, preactivation
        else:
            return output


class SoftmaxMLP(MLP):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes,
                 init_w=3e-3,
                 hidden_activation=F.leaky_relu,
                 output_activation=identity,
                 hidden_init=fanin_init,
                 b_init_value=0.1,
                 layer_norm=False,
                 layer_norm_params=None,
                 ):
        super(SoftmaxMLP, self).__init__(input_size,
                                         output_size,
                                         hidden_sizes,
                                         init_w,
                                         hidden_activation,
                                         output_activation,
                                         hidden_init,
                                         b_init_value,
                                         layer_norm,
                                         layer_norm_params)

    def forward(self, input, return_preactivation=False):
        output = super().forward(input, return_preactivation)
        if return_preactivation:
            output[0] = F.softmax(output[0], dim=-1)
        else:
            output = F.softmax(output, dim=-1)
        return output

    def sample(self, input, return_preactivation=False):
        probs = self.forward(input, return_preactivation)
        dist = D.Categorical(probs=probs)
        x = dist.sample()
        prob_z = dist.log_prob(x)
        return x, prob_z


class MultiSoftmaxMLP(SoftmaxMLP):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes,
                 head_num,
                 init_w=3e-3,
                 hidden_activation=F.leaky_relu,
                 output_activation=identity,
                 hidden_init=fanin_init,
                 b_init_value=0.1,
                 layer_norm=False,
                 layer_norm_params=None,
                 ):
        super(MultiSoftmaxMLP, self).__init__(input_size,
                                              output_size,
                                              hidden_sizes,
                                              init_w,
                                              hidden_activation,
                                              output_activation,
                                              hidden_init,
                                              b_init_value,
                                              layer_norm,
                                              layer_norm_params)
        self.head_num = head_num

    def forward(self, inputs, return_preactivation=False):
        output = super().forward(inputs, return_preactivation)
        if return_preactivation:
            output[0] = F.softmax(output[0], dim=-1)
        else:
            output = output.view(-1, self.head_num)
            output = F.softmax(output, dim=-1)
        return output

    def sample(self, input, return_preactivation=False, get_max=False):
        probs = self.forward(input, return_preactivation)
        if get_max:
            x = probs.argmax(dim=-1)
            prob_x = probs.max(dim=-1)
            log_prob_x = torch.log(prob_x)
        else:
            dist = D.Categorical(probs=probs)
            x = dist.sample()
            log_prob_x = dist.log_prob(x)
            prob_x = torch.exp(log_prob_x)
        return x, log_prob_x, prob_x

class MultiMLP(SoftmaxMLP):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes,
                 head_num,
                 init_w=3e-3,
                 hidden_activation=F.leaky_relu,
                 output_activation=identity,
                 hidden_init=fanin_init,
                 b_init_value=0.1,
                 layer_norm=False,
                 layer_norm_params=None,
                 ):
        super(MultiMLP, self).__init__(input_size,
                                              output_size,
                                              hidden_sizes,
                                              init_w,
                                              hidden_activation,
                                              output_activation,
                                              hidden_init,
                                              b_init_value,
                                              layer_norm,
                                              layer_norm_params)
        self.head_num = head_num

    def forward(self, input, return_preactivation=False):
        output = super().forward(input, return_preactivation)
        if return_preactivation:
            output[0] = F.softmax(output[0], dim=-1)
        else:
            output = output.view(-1, self.head_num)
            output = F.softmax(output, dim=-1)
        return output

    def sample(self, input, return_preactivation=False, get_max=False):
        probs = self.forward(input, return_preactivation)
        if get_max:
            x = probs.argmax(dim=-1)
            prob_x = probs.max(dim=-1)
        else:
            dist = D.Categorical(probs=probs)
            x = dist.sample()
            prob_x = dist.log_prob(x)
        return x, prob_x


class FlattenMLP(MLP):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MLPEncoder(FlattenMLP):
    def reset(self, num_task=1):
        pass
