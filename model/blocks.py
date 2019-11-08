'''
@Description: Blocks
@Author: xieydd
@Date: 2019-09-26 16:20:52
@LastEditTime: 2019-10-13 18:26:10
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.parameter import Parameter

class ShuffleChannels(nn.Module):
    def __init__(self, mid_channel,groups=2, **kwargs):
        super(ShuffleChannels, self).__init__()
        # For ShuffleNet v2, groups is always set 2
        assert groups == 2
        self.groups = groups
        self.mid_channel = mid_channel

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // self.groups, self.groups, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(self.groups, -1, num_channels // self.groups, height, width)
        return x[0], x[1]

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class ShuffleNetBlock(nn.Module):

    def __init__(self, inp, oup, mid_channels, ksize, stride, block_mode='ShuffleNetV2', bn=nn.BatchNorm2d, use_se=False, fix_arch=True, act_name='relu', shuffle_method=ShuffleChannels):
        super(ShuffleNetBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert block_mode in ['ShuffleNetV2', 'ShuffleXception']

        self.base_mid_channel = mid_channels
        self.ksize = ksize
        self.block_mode = block_mode
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        # project_input_C == project_mid_C == project_output_C == main_input_channel
        self.project_channel = inp // 2 if stride == 1 else inp
        # stride 1, input will be split
        self.main_input_channel = inp // 2 if stride == 1 else inp
        self.fix_arch = fix_arch

        self.outputs = oup - self.project_channel

        """
        Regular block: (We usually have the down-sample block first, then followed by repeated regular blocks)
        Input[64] -> split two halves -> main branch: [32] --> mid_channels (final_output_C[64] // 2 * scale[1.4])
                        |                                       |--> main_out_C[32] (final_out_C (64) - input_C[32]
                        |
                        |-----> project branch: [32], do nothing on this half
        Concat two copies: [64 - 32] + [32] --> [64] for final output channel

        =====================================================================

        In "Single path one shot nas" paper, Channel Search is searching for the main branch intermediate #channel.
        And the mid channel is controlled / selected by the channel scales (0.2 ~ 2.0), calculated from:
            mid channel = block final output # channel // 2 * scale

        Since scale ~ (0, 2), this is guaranteed: main mid channel < final output channel
        """
        self.channel_shuffle_and_split = shuffle_method(mid_channel=inp, groups=2)
        if block_mode == 'ShuffleNetV2':
            branch_main = [
                # pw
                nn.Conv2d(self.main_input_channel, self.base_mid_channel, 1, 1, 0, bias=False),
            ]

            if not self.fix_arch:
                branch_main += [ChannelSelector(channel_number=self.base_mid_channel)]

            branch_main += [
                bn(self.base_mid_channel),
                Activation(act_name),
                # dw
                nn.Conv2d(self.base_mid_channel, self.base_mid_channel, self.ksize, self.stride, self.pad,
                            groups=self.base_mid_channel, bias=False),
                bn(self.base_mid_channel),
                # pw-linear
                nn.Conv2d(self.base_mid_channel, self.outputs, 1, 1, 0, bias=False),
                bn(self.outputs),
                Activation(act_name)
            ]
        elif block_mode == 'ShuffleXception':
            branch_main = [
                # dw
                nn.Conv2d(self.main_input_channel,self.main_input_channel, self.ksize, self.stride, self.pad, groups=self.main_input_channel, bias=False),
                bn(self.main_input_channel),
                # pw
                nn.Conv2d(self.main_input_channel,self.base_mid_channel, 1, 1, 0, bias=False),
            ]

            if not self.fix_arch:
                branch_main += [ChannelSelector(channel_number=self.base_mid_channel)]
            
            branch_main += [
                bn(self.base_mid_channel),
                Activation(act_name),
                # dw
                nn.Conv2d(self.base_mid_channel, self.base_mid_channel, self.ksize, 1, self.pad, groups=self.base_mid_channel, bias=False),
                bn(self.base_mid_channel),
                # pw
                nn.Conv2d(self.base_mid_channel, self.base_mid_channel, 1, 1, 0, bias=False),
            ]
            
            if not self.fix_arch:
                branch_main += [ChannelSelector(channel_number=self.base_mid_channel)]

            branch_main += [
                bn(self.base_mid_channel),
                Activation(act_name),
                # dw
                nn.Conv2d(self.base_mid_channel, self.base_mid_channel, self.ksize, 1, self.pad, groups=self.base_mid_channel, bias=False),
                bn(self.base_mid_channel),
                # pw
                nn.Conv2d(self.base_mid_channel, self.outputs, 1, 1, 0, bias=False),
                bn(self.outputs),
                Activation(act_name),
            ]
        if use_se:
            branch_main += [SE(self.outputs)]
        if fix_arch:
            self.branch_main = nn.Sequential(*branch_main)
        else:
            #base = NasBaseHybridSequential(*branch_main)
            #self.branch_main = nn.Sequential(*[base.append(a) for a in branch_main])
            self.branch_main = NasBaseHybridSequential(branch_main) 
        if stride == 2:
            """
            Down-sample block:
            Input[16] -> two copies -> main branch: [16] --> mid_channels (final_output_C[64] // 2 * scale[1.4])
                            |                                   |--> main_out_C[48] (final_out_C (64) - input_C[16])
                            |
                            |-----> project branch: [16] --> project_mid_C[16] --> project_out_C[16]
            Concat two copies: [64 - 16] + [16] --> [64] for final output channel
            """
            branch_proj = [
                # dw
                nn.Conv2d(self.project_channel, self.project_channel, self.ksize, self.stride, self.pad, groups=self.project_channel, bias=False),
                bn(self.project_channel),
                # pw-linear
                nn.Conv2d(self.project_channel, self.project_channel, 1, 1, 0, bias=False),
                bn(self.project_channel),
                Activation(act_name),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, x):
        if self.stride==2:
            x_proj = x
            x_main = x
            # print("strides == 2 ShuffleNetBlock")
            # print(x_main.shape)
            # print((self.branch_proj(x_proj)).shape)
            # print((self.branch_main(x_main)).shape)
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x_main)), 1)
        elif self.stride==1:
            x_proj, x_main = self.channel_shuffle_and_split(x)
            # print("stride == 1 ShuffleNetBlock")
            # print(x_proj.shape)
            # print(x_main.shape)
            # print((self.branch_main(x_main)).shape)
            return torch.cat((x_proj, self.branch_main(x_main)), 1)

class ShuffleNasBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride,bn=nn.BatchNorm2d, max_channel_scale=2.0, 
                 use_all_blocks=False, act_name='relu', use_se=False, **kwargs):
        super(ShuffleNasBlock, self).__init__()
        assert stride in [1, 2]
        self.use_all_blocks = use_all_blocks
        """
        Four pre-defined blocks
        """
        max_mid_channel = int(output_channel // 2 * max_channel_scale)
        self.block_sn_3x3 = ShuffleNetCSBlock(input_channel, output_channel, max_mid_channel,
                                                3, stride, 'ShuffleNetV2', bn=bn,act_name=act_name, use_se=use_se)
        self.block_sn_5x5 = ShuffleNetCSBlock(input_channel, output_channel, max_mid_channel,
                                                5, stride, 'ShuffleNetV2', bn=bn,act_name=act_name, use_se=use_se)
        self.block_sn_7x7 = ShuffleNetCSBlock(input_channel, output_channel,max_mid_channel,
                                                7, stride, 'ShuffleNetV2', bn=bn,act_name=act_name, use_se=use_se)
        self.block_sx_3x3 = ShuffleNetCSBlock(input_channel, output_channel,max_mid_channel,
                                                  3, stride, 'ShuffleXception', bn=bn,act_name=act_name, use_se=use_se)

    def forward(self, x, block_choice, block_channel_mask, *args, **kwargs):
        # ShuffleNasBlock has three inputs and passes two inputs to the ShuffleNetCSBlock
        if self.use_all_blocks:
            temp1 = self.block_sn_3x3(x, block_channel_mask)
            temp2 = self.block_sn_5x5(x, block_channel_mask)
            temp3 = self.block_sn_7x7(x, block_channel_mask)
            temp4 = self.block_sx_3x3(x, block_channel_mask)
            x = temp1 + temp2 + temp3 + temp4
        else:
            if block_choice == 0:
                x = self.block_sn_3x3(x, block_channel_mask)
            elif block_choice == 1:
                x = self.block_sn_5x5(x, block_channel_mask)
            elif block_choice == 2:
                x = self.block_sn_7x7(x, block_channel_mask)
            elif block_choice == 3:
                x = self.block_sx_3x3(x, block_channel_mask)
        return x


class ShuffleNetCSBlock(ShuffleNetBlock):
    """
    ShuffleNetBlock with Channel Selecting
    """
    def __init__(self, input_channel, output_channel,mid_channel, ksize, stride,
                 block_mode='ShuffleNetV2', bn=nn.BatchNorm2d, fix_arch=False, act_name='relu', use_se=False, **kwargs):
        super(ShuffleNetCSBlock, self).__init__(input_channel, output_channel, mid_channel, ksize, stride,
                                                block_mode=block_mode, use_se=use_se, fix_arch=fix_arch,
                                                act_name=act_name, **kwargs)

    def forward(self, x, channel_choice):
        #self.branch_main.cuda()
        if self.stride == 2:
            x_project = x
            x_main = x
            #print("strides == 2 ShuffleNetCSBlock")
            #print("x_main shape", x_main.shape)
            #print("branch_proj shape", (self.branch_proj(x)).shape)
            #print("branch_main shape", (self.branch_main(x_main, channel_choice)).shape)
            #self.branch_proj.cuda()
            return torch.cat((self.branch_proj(x_project), self.branch_main(x_main, channel_choice)), dim=1)
        elif self.stride == 1:
            #print("strides == 1 ShuffleNetCSBlock")
            #print(x.shape)
            x_project, x_main = self.channel_shuffle_and_split(x)
            return torch.cat((x_project, self.branch_main(x_main, channel_choice)), dim=1)

class NasBaseHybridSequential(nn.Module):
    def __init__(self, blocks):
        super(NasBaseHybridSequential, self).__init__()
        self.blocks = blocks

    def forward(self, x, block_channel_mask):
        for block in self.blocks:
            #print("NasBaseHybridSequential")
            #print(block)
            #print(x.shape)
            #print(block_channel_mask.shape)
            #block.cuda()
            if isinstance(block, ChannelSelector):
                x = block(x, block_channel_mask)
            else:
                x = block(x)
        return x

class ShuffleChannelsConv(nn.Module):
    """
    ShuffleNet channel shuffle Block.
    """
    def __init__(self, mid_channel, groups=2, **kwargs):
        super(ShuffleChannelsConv, self).__init__()
        # For ShuffleNet v2, groups is always set 2
        assert groups == 2
        self.groups = groups
        self.mid_channel = int(mid_channel)
        self.channels = int(mid_channel * 2)
        self.transpose_conv = nn.Conv2d(self.channels, self.channels, 1, 1,padding=0, bias=False)

    # def transpose_init(self):
    #     for i, param in enumerate(self.transpose_conv.collect_params().values()):
    #         if i > 0:
    #             raise ValueError('Transpose conv should only have the weights parameter.')
    #         param.set_data(self.generate_transpose_conv_kernel())
    #         param.grad_req = 'null'

    # def generate_transpose_conv_kernel(self):
    #     c = self.channels
    #     if c % 2 != 0:
    #         raise ValueError('Channel number should be even.')
    #     idx = np.zeros(c)
    #     idx[np.arange(0, c, 2)] = np.arange(c / 2)
    #     idx[np.arange(1, c, 2)] = np.arange(c / 2, c, 1)
    #     weights = np.zeros((c, c))
    #     weights[np.arange(c), idx.astype(int)] = 1.0
    #     print(weights)
    #     return nd.expand_dims(nd.expand_dims(nd.array(weights), axis=2), axis=3)

    def forward(self, x):
        data = self.transpose_conv(x)
        data_project = data.narrow(1,0,self.mid_channel)
        #print(self.mid_channel)
        data_x = data.narrow(1,self.mid_channel,data.shape[1] - self.mid_channel)
        return data_project, data_x


class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class Hard_Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hard_Sigmoid, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        x = 0.2 * x + 0.5
        if self.inplace:
            return x.clamp_(0, 1)
        else:
            return x.clamp(0, 1)

class Activation(nn.Module):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    Returns
    -------
    nn.Module
        Activation layer.
    """
    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif activation == "swish":
            self.act = Swish()
        elif activation == "hswish":
            self.act = HSwish(inplace=True)
        elif activation == "hard_sigmoid":
            self.act = Hard_Sigmoid()
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.act(x)

class SE(nn.Module):
    def __init__(self, num_in, ratio=4,
                 act_func=("relu", "hard_sigmoid"), use_bn=False, **kwargs):
        super(SE, self).__init__(**kwargs)
    
        def make_divisible(x, divisible_by=8):
                # make the mid channel to be divisible to 8 can increase the cache hitting ratio from SENET and descrise the computer cost and parameters numbers
                return int(np.ceil(x * 1. / divisible_by) * divisible_by)

        self.use_bn = use_bn
        num_out = num_in
        num_mid = make_divisible(num_out // ratio)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_in, num_mid, 1, bias=True),
            Activation(act_func[0]),
            nn.Conv2d(num_mid, num_out, 1, bias=True),
            Activation(act_func[1])
        )
    def forward(self, x):
        out = self.channel_attention(x)
        return x * out

class ChannelSelector(nn.Module):
    """
    Random channel # selection
    """
    def __init__(self, channel_number, candidate_scales=None):
        super(ChannelSelector, self).__init__()
        if candidate_scales is None:
            self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        else:
            self.candidate_scales = candidate_scales
        self.channel_number = channel_number

    def forward(self, x, block_channel_mask, *args, **kwargs):
        #indices = torch.LongTensor([0, self.channel_number])
        #block_channel_mask = torch.index_select(block_channel_mask, -1,indices)
        block_channel_mask = block_channel_mask.narrow(1, 0,self.channel_number)
        # block_channel_mask = np.array(block_channel_mask[0][:self.channel_number]).reshape(1, self.channel_number, 1, 1) # if use tensor will split in dataparallel
        #print(block_channel_mask.shape)
        block_channel_mask = block_channel_mask.reshape(shape=(1, self.channel_number, 1, 1))
        device = torch.device('cuda')
        #block_channel_mask = block_channel_mask.to(device)
        x = x * block_channel_mask
        # x = x * torch.Tensor(channel_channel_mask).cuda() #if use tensor will split in dataparallel
        return x

class NasHybridSequential(nn.Module):
    def __init__(self, blocks):
        super(NasHybridSequential, self).__init__()
        self.blocks = blocks
    def forward(self, x, full_arch, full_channel_mask):
        nas_index = 0
        base_index = 0
        for block in self.blocks:
            if isinstance(block, ShuffleNasBlock):
                block_choice = full_arch[nas_index]
                block_channel_mask = full_channel_mask[nas_index:nas_index + 1]
                x = block(x, block_choice, block_channel_mask)
                nas_index += 1
            elif isinstance(block, ShuffleNetBlock):
                block_channel_mask = full_channel_mask[base_index: base_index + 1]
                x = block(x, block_channel_mask)
                base_index += 1
            else:
                x = block(x)
        # assert (nas_index == full_arch.shape[0] == full_channel_mask.shape[0] or
        #         base_index == full_arch.shape[0] == full_channel_mask.shape[0])
        return x


# https: // github.com/megvii-model/ShuffleNet-Series/issues/9
class NasBatchNorm(nn.modules.batchnorm._BatchNorm):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, inference_update_stat=False):
        super(nn.modules.batchnorm._BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.inference_update_stat = inference_update_stat
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.inference_update_stat:
            mean = torch.mean(input, dim=[0,2,3])
            mean_expanded = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mean, dim=0), dim=2),dim=3)
            var = torch.mul(x*mean_expanded, x*mean_expanded).mean(dim=[0, 2, 3])


            self.running_mean = torch.add(torch.mul(self.running_mean, self.momentum), torch.mul(mean, self.momentum))
            self.running_var = torch.add(torch.mul(self.running_var, self.momentum), torch.mul(var, self.momentum))
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(
                   **self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(
                    0, dtype=torch.long)

# add flops constraint
def random_block_choices(stage_repeats=None, num_of_block_choices=4,timeout=500):
    if stage_repeats is None:
        stage_repeats = [4, 4, 8, 4]
    block_number = sum(stage_repeats)

    # flosp constraint
    flops_l, flops_r, flops_step = 290, 360, 10
    bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]
    idx = np.random.randint(len(bins))
    l, r = bins[idx]
    for i in range(timeout):
        block_choices = []
        for i in range(block_number):
            block_choices.append(random.randint(0, num_of_block_choices - 1))
        if l*1e6 <= get_cand_flops(block_choices) <= r*1e6:
            return torch.Tensor(block_choices)
    return random_block_choices()

def random_channel_mask(stage_repeats=None, stage_out_channels=None, candidate_scales=None,
                        select_all_channels=False):
    """
    candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    """
    if stage_repeats is None:
        stage_repeats = [4, 4, 8, 4]
    if stage_out_channels is None:
        stage_out_channels = [64, 160, 320, 640]
    if candidate_scales is None:
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    assert len(stage_repeats) == len(stage_out_channels)

    channel_mask = []
    global_max_length = int(stage_out_channels[-1] // 2 * candidate_scales[-1])
    for i in range(len(stage_out_channels)):
        local_max_length = int(stage_out_channels[i] // 2 * candidate_scales[-1])
        local_min_length = int(stage_out_channels[i] // 2 * candidate_scales[0])
        for _ in range(stage_repeats[i]):
            if select_all_channels:
                local_mask = [1] * global_max_length
            else:
                local_mask = [0] * global_max_length
                random_select_channel = random.randint(local_min_length, local_max_length)
                for j in range(random_select_channel):
                    local_mask[j] = 1
            channel_mask.append(local_mask)
    return torch.Tensor(channel_mask)


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    """Test ShuffleNetCSBlock"""
    block_mode = 'ShuffleXception'
    dummy = torch.rand(5, 4, 224, 224)
    cs_block = ShuffleNetCSBlock(input_channel=4, output_channel=16, mid_channel=16, ksize=3, 
                                 stride=1, block_mode=block_mode)
    
    #print(cs_block)
    # generate channel mask
    channel_mask = torch.Tensor([[1] * 10 + [0] * 6])
    import torch.autograd as ag
    rst = cs_block(dummy, channel_mask)
    rst.backward(torch.ones_like(rst))
    params = cs_block.parameters()
    for param in params:
        print(param.name)

    """ Test ShuffleChannels """
    channel_shuffle = ShuffleChannels(mid_channel=4, groups=2)
    s = torch.ones([1, 8, 3, 3])
    s[:, 4:, :, :] *= 2
    s_project, s_main = channel_shuffle(s)
    print(s)
    print(s_project)
    print(s_main)
    print("Finished testing ShuffleChannels\n")
    
    """ Test ShuffleBlock with "ShuffleNetV2" mode """
    tensor = torch.ones([1, 4, 14, 14])
    tensor[:, 2:, :, :] = 2
    block0 = ShuffleNetBlock(inp=4, oup=16,
                             mid_channels=int(16 // 2 * 1.4), ksize=3, stride=2, block_mode='ShuffleNetV2')
    block1 = ShuffleNetBlock(inp=16, oup=16,
                             mid_channels=int(16 // 2 * 1.2), ksize=3, stride=1, block_mode='ShuffleNetV2')
    temp0 = block0(tensor)
    temp1 = block1(temp0)
    print(temp0.shape)
    print(temp1.shape)
    # print(block0)
    # print(block1)
    print("Finished testing ShuffleNetV2 mode\n")

    """ Test ShuffleBlock with "ShuffleXception" mode """
    tensor = torch.ones([1, 4, 14, 14])
    tensor[:, 2:, :, :] = 2
    blockx0 = ShuffleNetBlock(inp=4, oup=16,
                              mid_channels=int(16 // 2 * 1.4), ksize=3, stride=2, block_mode='ShuffleXception')
    blockx1 = ShuffleNetBlock(inp=16, oup=16,
                              mid_channels=int(16 // 2 * 1.2), ksize=3, stride=1, block_mode='ShuffleXception')
    tempx0 = blockx0(tensor)
    tempx1 = blockx1(temp0)
    print(tempx0.shape)
    print(tempx1.shape)
    # print(blockx0)
    # print(blockx1)
    print("Finished testing ShuffleXception mode\n")

    """ Test ChannelSelection """
    block_final_output_channel = 8
    candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    max_channel = int(block_final_output_channel // 2 * candidate_scales[-1])
    tensor = torch.ones([1, max_channel, 14, 14])
    for i in range(max_channel):
        tensor[:, i, :, :] = i
    channel_selector = ChannelSelector(channel_number=block_final_output_channel)
    for i in range(4):
        global_channel_mask = random_channel_mask(stage_out_channels=[8, 160, 320, 640])
        #indices = torch.LongTensor([i])
        #local_channel_mask = torch.index_select(global_channel_mask, 1, indices)
        local_channel_mask = global_channel_mask[i:i+1]
        #local_channel_mask = nd.slice(global_channel_mask, begin=(i, None), end=(i + 1, None))
        selected_tensor = channel_selector(tensor, local_channel_mask)
        print(selected_tensor.shape)
    print("Finished testing ChannelSelector\n")

    """ Test BN with inference statistic update """
    # bn = NasBatchNorm(inference_update_stat=True, in_channels=4)
    
    # mean, std = 5, 2
    # for i in range(100):
    #     dummy = torch.normal(torch.Tensor(mean), torch.Tensor(std), out=torch.Tensor([10, 4, 5, 5]))
    #     rst = bn(dummy)
    #     # print(dummy)
    #     # print(rst)
    # bn.running_mean.set_data(bn.running_mean.data() + 1)
    # print("Defined mean: {}, running mean: {}".format(mean, bn.running_mean.data()))
    # print("Defined std: {}, running var: {}".format(std, bn.running_var.data()))
    # print("Finished testing NasBatchNorm\n")

    dummy = np.ones((1,6,5,5))
    for i in range(6):
        dummy[:,i,:,:] = i

    transpose_conv = ShuffleChannelsConv(mid_channel=6/2)
    data_project, data_x = transpose_conv(torch.Tensor(dummy))
    print(data_project.shape)
    print(data_x.shape)
    
if __name__ == "__main__":
    main()
