import math
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch import nn

def surround_kernel(sigma=2):
    """
    The output kernel size is int(8 * sigma + 1)
    :param sigma:
    :return:
    """
    ax = np.linspace(int(-4 * sigma), int(4 * sigma), int(8 * sigma + 1))
    x, y = np.meshgrid(ax, ax)
    g1_sigma = 4 * sigma
    g2_sigma = sigma
    g1 = np.exp(-(x ** 2 + y ** 2) / (2 * g1_sigma ** 2)) / (2 * np.pi * g1_sigma ** 2)
    g2 = np.exp(-(x ** 2 + y ** 2) / (2 * g2_sigma ** 2)) / (2 * np.pi * g2_sigma ** 2)
    g = g1 - g2
    g[g < 0] = 0
    g = g / np.sum(g)
    return g


class PushPullConv2DUnit(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            avg_kernel_size: _size_2_t,
            pull_inhibition_strength: int = 1,
            trainable_pull_inhibition: bool = False,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None,
            use_attn=False):

        super(PushPullConv2DUnit, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.out_channels = out_channels
        self.trainable_pull_inhibition = trainable_pull_inhibition

        if trainable_pull_inhibition:
            self.pull_inhibition_strength = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.pull_inhibition_strength.data.uniform_(0, 1)
        else:
            self.pull_inhibition_strength = pull_inhibition_strength

        self.push_conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode, device=device,
            dtype=dtype)
        
        # Attention mechanism
        self.use_attn = use_attn
        if self.use_attn:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        # ss_kernel_2d = surround_kernel(sigma=2)
        # ss_kernel = np.zeros((self.out_channels, self.out_channels, *ss_kernel_2d.shape))
        # import pdb; pdb.set_trace()
        # for index in range(ss_kernel_2d.shape[0]):
            # ss_kernel[index, index] = ss_kernel_2d
        # self.surround_kernel = torch.tensor(ss_kernel, device=device).to(torch.float32)

        if avg_kernel_size != 0:
            self.avg = torch.nn.AvgPool2d(
                kernel_size=avg_kernel_size,
                stride=1,
                padding=tuple([int((x - 1) / 2) for x in _pair(avg_kernel_size)]),
                count_include_pad=False
            )
        else:
            self.avg = None

        # pooling = gauss_kernel(kernel_size[0], sig=0.25)
        # pooling = pooling / np.max(pooling)
        # pooling = 1 - pooling
        # self.pooling = torch.tensor(pooling, device=device, dtype=dtype)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.bias.data.uniform_(-1, 1)  # random weight initialization
        else:
            self.bias = None

    @property
    def weight(self):
        return self.push_conv.weight

    @weight.setter
    def weight(self, value):
        self.push_conv.weight = value

    def forward_surround_suppression(self, x):
        """
        This is an implementation of surround-suppression. The idea is to inhibit the push response w.r.t its surround.
        This operation would suppress response to noise and fine-texture (for example, a single blade of grass).
        Features with high-contrast, especially high-level shapes would be preserved. Thereby introducing bias towards
        shape based features during the process of feature extraction. We believe, this bias would introduce robustness
        towards high-level shape based features.
        :param x: 4D input batch - (batch, channels, height, width)
        :return: surround-suppressed response of the convolution
        """

        # Compute the push-response
        push_response = self.push_conv(x)
        push_response = F.relu_(push_response)

        # Surround response
        surr_response = F.conv2d(push_response, self.surround_kernel, padding='same')

        # Suppressed push-response
        if not self.trainable_pull_inhibition:
            x_out = push_response - surr_response * self.pull_inhibition_strength
        else:
            x_out = push_response - surr_response * self.pull_inhibition_strength.view((1, -1, 1, 1))

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        return x_out

    def forward(self, x):
        """
        PushPull based inhibition
        :param x:
        :return:
        """
        push_kernel = self.push_conv.weight
        push_min = torch.amin(push_kernel, dim=(1, 2, 3), keepdim=True)
        push_max = torch.amax(push_kernel, dim=(1, 2, 3), keepdim=True)
        pull_kernel = -push_kernel + (push_max + push_min)
        # push_sum = torch.sum(push_kernel, dim=(1, 2, 3), keepdims=True)
        # pull_sum = torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True)

        # eps = torch.finfo(torch.float32).eps
        # pull_kernel = pull_kernel / (torch.abs(pull_sum) + eps) * (torch.abs(push_sum) + eps)
        # inhibition_sign = torch.sign(push_sum) / torch.sign(pull_sum)
        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        if self.avg:
            pull_response = self.avg(pull_response)
        push_response = F.relu_(push_response)
        pull_response = F.relu_(pull_response)

        ## Apply Attention to push kernels
        if self.use_attn:
            attention_weights = self.attention(push_response)
            push_response = push_response * attention_weights

        if not self.trainable_pull_inhibition:
            x_out = push_response - pull_response * self.pull_inhibition_strength
        else:
            x_out = push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1))

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        return x_out

    def forward_dev(self, x):
        push_kernel = self.push_conv.weight
        min_push = torch.amin(push_kernel, dim=(2, 3), keepdim=True)
        max_push = torch.amax(push_kernel, dim=(2, 3), keepdim=True)
        pull_kernel = -push_kernel + (max_push + min_push)
        push_sum = torch.sum(push_kernel, dim=(2, 3), keepdims=True)
        pull_sum = torch.sum(pull_kernel, dim=(2, 3), keepdims=True)
        pull_kernel = pull_kernel / pull_sum * push_sum

        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        if self.avg:
            pull_response = self.avg(pull_response)

        push_response = F.relu_(push_response)
        pull_response = F.relu_(pull_response)

        x_out = push_response - pull_response

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))

        return x_out

    def _forward_cvpr(self, x):
        # plot_data = [('input', x)]
        W = self.push_conv.weight
        min_push = torch.amin(W, dim=(1, 2, 3), keepdim=True)
        max_push = torch.amax(W, dim=(1, 2, 3), keepdim=True)
        pull_kernel = -W + (max_push + min_push)
        # pull_kernel = self.get_pull_kernel(W, pull_kernel)

        # z = (W - W.mean(dim=(1, 2, 3), keepdims=True)) / W.std(dim=(1, 2, 3), keepdims=True)
        # max_std = 2
        # min_push = torch.amin(torch.where(torch.logical_and(z > -max_std, z < max_std), W, torch.inf),
        #                       dim=(1, 2, 3), keepdim=True)
        # max_push = torch.amax(torch.where(torch.logical_and(z > -max_std, z < max_std), W, -torch.inf),
        #                       dim=(1, 2, 3), keepdim=True)
        # pull_kernel = -W + (max_push + min_push)

        # push_sum = torch.abs(torch.sum(W, dim=(1, 2, 3), keepdims=True))
        # pull_sum = torch.abs(torch.sum(pull_kernel, dim=(1, 2, 3), keepdims=True))
        # pull_kernel = pull_kernel / pull_sum * push_sum
        # pull_kernel[:32] = self.normalize_pull_kernel(W[:32], pull_kernel[:32])

        push_response = self.push_conv(x)
        pull_response = F.conv2d(x, pull_kernel, None, self.stride, self.padding, self.dilation, self.groups)
        # pull_response = self.pull_conv(x)

        if self.avg:
            pull_response = self.avg(pull_response)
        # plot_data.extend([('push_response', push_response), ('pull_response', pull_response)])

        push_response = F.relu_(push_response)
        pull_response = F.relu_(pull_response)
        # plot_data.extend([('push_response + ReLU', push_response), ('pull_response + ReLU', pull_response)])

        if not self.trainable_pull_inhibition:
            x_out = push_response - pull_response * self.pull_inhibition_strength
        else:
            x_out = push_response - pull_response * self.pull_inhibition_strength.view((1, -1, 1, 1))
        # plot_data.extend([('x_out', x_out)])

        if self.bias is not None:
            x_out = x_out + self.bias.view((1, -1, 1, 1))
        # plot_data.extend([('x_out + bias', x_out)])

        # plot_minibatch_inputs(x)
        # plot_push_kernels(self.push_conv.weight)
        # plot_intermediate_response(plot_data, img_index=0, filters_to_plot=(9,))

        return x_out

def plot_intermediate_response(plot_data, img_index=0, filters_to_plot=(0,)):
    # bring all tensors from GPU to CPU
    plot_data_cpu = [(name, tensor[img_index].cpu().detach().numpy()) for name, tensor in plot_data]

    # plot attributes
    num_rows = 2
    num_cols = math.ceil(len(plot_data_cpu) / num_rows)

    for filter_id in filters_to_plot:
        fig, ax = plt.subplots(num_rows, num_cols, dpi=200, figsize=(num_cols * 2, num_rows * 2))
        row_id, col_id = 0, 0
        for name, tensor in plot_data_cpu:
            if name == 'input':
                arr_to_plot = np.transpose(tensor, axes=[1, 2, 0])  # channels last
            else:
                arr_to_plot = tensor[filter_id, :, :]
            img = ax[row_id][col_id].imshow(arr_to_plot)

            divider = make_axes_locatable(ax[row_id][col_id])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(img, cax=cax, orientation='vertical')
            ax[row_id][col_id].set_title(name)
            ax[row_id][col_id].axis('off')

            col_id = (col_id + 1) % num_cols
            if col_id == 0:
                row_id += 1

        plt.tight_layout()
        plt.show()
        plt.close()


def plot_minibatch_inputs(plot_data):
    plot_data_cpu = [tensor.cpu().detach().numpy() for tensor in plot_data]
    num_rows = num_cols = math.ceil(math.sqrt(len(plot_data_cpu)))
    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, dpi=200, figsize=(15, 15))
    row_id, col_id = 0, 0

    for idx, tensor in enumerate(plot_data_cpu):
        arr_to_plot = np.transpose(tensor, axes=[1, 2, 0])  # channels last
        ax[row_id][col_id].imshow(arr_to_plot)
        ax[row_id][col_id].set_title(f'{idx}')

        col_id = (col_id + 1) % num_cols
        if col_id == 0:
            row_id += 1

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_push_kernels(plot_data, title=None):
    plot_data_cpu = [tensor.cpu().detach().numpy() for tensor in plot_data]
    num_rows = num_cols = math.ceil(math.sqrt(len(plot_data_cpu)))
    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(15, 7), constrained_layout=True)
    row_id, col_id = 0, 0
    for idx, tensor in enumerate(plot_data_cpu):
        # arr_to_plot = np.transpose(tensor, axes=[1, 2, 0])  # channels last
        img = ax[row_id][col_id].imshow(np.concatenate([tensor[0], tensor[1], tensor[2]], axis=1))
        divider = make_axes_locatable(ax[row_id][col_id])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')
        ax[row_id][col_id].set_title(f'{idx}')
        col_id = (col_id + 1) % num_cols
        if col_id == 0:
            row_id += 1
    # if title:
    #     plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()