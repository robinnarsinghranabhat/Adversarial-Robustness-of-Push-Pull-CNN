import torch
import math
from torch import nn
import torch.nn.functional as F


class PPmodule2d(nn.Module):
    """
    Implementation of the Push-Pull layer from:
    [1] N. Strisciuglio, M. Lopez-Antequera, N. Petkov,
    Enhanced robustness of convolutional networks with a push–pull inhibition layer,
    Neural Computing and Applications, 2020, doi: 10.1007/s00521-020-04751-8

    It is an extension of the Conv2d module, with extra arguments:

    * :attr:`alpha` controls the weight of the inhibition. (default: 1 - same strength as the push kernel)
    * :attr:`scale` controls the size of the pull (inhibition) kernel (default: 2 - double size).
    * :attr:`dual_output` determines if the response maps are separated for push and pull components.
    * :attr:`train_alpha` controls if the inhibition strength :attr:`alpha` is trained (default: False).


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        alpha (float, optional): Strength of the inhibitory (pull) response. Default: 1
        scale (float, optional): size factor of the pull (inhibition) kernel with respect to the pull kernel. Default: 2
        dual_output (bool, optional): If ``True``, push and pull response maps are places into separate channels of the output. Default: ``False``
        train_alpha (bool, optional): If ``True``, set alpha (inhibition strength) as a learnable parameters. Default: ``False``
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 alpha=1, scales=[1.5, 2, 3], dual_output=False,
                 train_alpha=False,
                 use_attn=True):
        super(PPmodule2d, self).__init__()

        self.train_alpha = train_alpha

        # Push kernels (is the one for which the weights are learned - the pull kernel is derived from it)
        self.push = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.scales = scales
        self.sharpens = nn.ModuleList([
            self._create_sharpen(scale) for scale in scales
        ])

        concat_output_channels = out_channels * (1 + len(scales))


        # Attention mechanism
        self.use_attn = use_attn
        if self.use_attn:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                # nn.Flatten(start_dim=1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=True),
                # nn.Linear(concat_output_channels, concat_output_channels // 4, bias=True),
                nn.GELU(),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=True),
                # nn.Linear(concat_output_channels // 4, concat_output_channels, bias=True),
                nn.Sigmoid()
            )

        # Channel reducer (1x1 convolution)
        self.channel_reducer = nn.Conv2d(
            in_channels=concat_output_channels,  # 4 channels per group (p, p1, p2, p3)
            out_channels=out_channels,
            kernel_size=1,
            groups=out_channels
        )


        # Configuration of the Push-Pull inhibition
        if not self.train_alpha:
            # when alpha is an hyper-parameter (as in [1])
            self.alpha = alpha
        else:
            # when alpha is a trainable parameter
            k = 1
            self.alpha = nn.Parameter(k * torch.ones(1, out_channels, 1, 1), requires_grad=True)
            r = 1. / math.sqrt(in_channels * out_channels)
            self.alpha.data.uniform_(.5-r, .5+r)  # math.sqrt(n) / 2)  # (-stdv, stdv)

        self.relu = nn.GELU()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        ## Removing Relu. to avoid information loss ! 
        push = self.push(x)        
        sharpen_activations = [ sharpen(x) for sharpen in self.sharpens]

        stacked = torch.stack([push] + sharpen_activations, dim=2)  # shape: (B, C, 4, H, W)
    
        # Reshape to interleave channels: [p[0], p1[0], p2[0], p3[0], p[1], p1[1], ...]
        concat_features = stacked.flatten(start_dim=1, end_dim=2)  # shape: (B, C*4, H, W) = (B, 64, 32, 32)
        
        output = self.channel_reducer(concat_features)  # shape: (B, 16, 32, 32)
        ## Apply Attention to push kernels
        if self.use_attn:
            attention_weights = self.attention(output)
            output = attention_weights * output

        # output = self.channel_reducer(concat_features)
        return output

    def _create_sharpen(self, scale):
        """
        Creates a sharpen layer for a given scale.
        """
        class SharpenLayer(nn.Module):
            def __init__(self, main_conv, scale):
                super().__init__()
                self.main_conv = main_conv
                self.scale = scale
                padding = 0

                # Compute the size of the sharpen kernel
                main_kernel_size = self.main_conv.weight.size()[2]  # Assuming square kernel
                if main_kernel_size % 2 == 0:
                    main_kernel_size += 1
                # input : 1.5, 2, 3 
                sharpen_size = math.floor(main_kernel_size * scale)
                # >> 4, 6, 9
                if sharpen_size % 2 == 0:
                    sharpen_size += 1 
                # 5, 7, 9
                self.sharpen_padding = ( sharpen_size  - 1 ) // 2 ## works for Stride = 1
                # 2 , 3, 4
                # sharpen_size = int(main_kernel_size * scale)
                # if sharpen_size % 2 == 0:
                #     sharpen_size += 1  # Ensure odd kernel size for symmetric padding

                # Upsampler to create the sharpen kernel
                self.up_sampler = nn.Upsample(size=(sharpen_size, sharpen_size), mode='bilinear', align_corners=True)

                # Compute padding to ensure output size matches main_conv's output
                # self.sharpen_padding = (sharpen_size - main_kernel_size) // 2

            def forward(self, x):
                # Upsample and negate the main_conv weights to create the sharpen kernel
                sharpen_weights = self.up_sampler(self.main_conv.weight)
                sharpen_weights = -sharpen_weights  # Negate the weights for sharpening

                # Apply the sharpen convolution
                sharpen_activations = F.conv2d(
                    x,
                    sharpen_weights,
                    bias=None,
                    stride=self.main_conv.stride,
                    padding=self.sharpen_padding,
                    dilation=self.main_conv.dilation,
                    groups=self.main_conv.groups
                )
                return sharpen_activations

        return SharpenLayer(self.push, scale)

