'''
@Description: ShuffleNet_OneShot
@Author: your name
@Date: 2019-09-26 19:44:29
@LastEditTime: 2019-10-11 22:45:14
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
from blocks import ShuffleNetBlock, Activation, ShuffleNasBlock, SE, NasHybridSequential, ShuffleChannels, ShuffleChannelsConv, GlobalAvgPool2d, NasBatchNorm
import random
from sys import maxsize
import numpy as np
from thop import profile
from thop import clever_format
from torchsummary import summary


class ShuffleNetV2_OneShot(nn.Module):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channels_scales=None,
                use_all_blocks=False, bn=nn.BatchNorm2d, use_se=False, last_conv_after_pooling=False,
                shuffle_method=ShuffleChannels, stage_out_channels=None, candidate_scales=None):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4,
            2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6,
            0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640,
            1024] if stage_out_channels is None else stage_out_channels
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] \
            if candidate_scales is None else candidate_scales
        self.use_all_blocks = use_all_blocks
        self.use_se = use_se
        self.last_conv_after_pooling = last_conv_after_pooling

        if architecture is None and channels_scales is None:
            fix_arch = False
        elif architecture is not None and channels_scales is not None:
            fix_arch = True
            assert len(architecture) == len(channels_scales)
        else:
            raise ValueError(
                "architecture and scale_ids should be both None or not None.")
        self.fix_arch = fix_arch
        assert len(self.stage_repeats) == len(self.stage_out_channels) - 3

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            bn(input_channel),
            Activation('hswish' if self.use_se else 'relu'),
        )

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            if self.use_se:
                act_name = 'hswish' if idxstage >= 1 else 'relu'
                block_use_se = True if idxstage >= 2 else False
            else:
                act_name = 'relu'
                block_use_se = False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                if fix_arch:
                    blockIndex = architecture[archIndex]
                    base_mid_channels = outp // 2
                    mid_channels = make_divisible(
                        int(base_mid_channels * channels_scales[archIndex]))
                    archIndex += 1
                    if blockIndex == 0:
                        # print('Shuffle3x3')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, bn=bn, ksize=3, stride=stride,
                                             block_mode='ShuffleNetV2', use_se=block_use_se, act_name=act_name, shuffle_method=shuffle_method))
                    elif blockIndex == 1:
                        # print('Shuffle5x5')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, bn=bn, ksize=5, stride=stride,
                                             block_mode='ShuffleNetV2', use_se=block_use_se, act_name=act_name, shuffle_method=shuffle_method))
                    elif blockIndex == 2:
                        # print('Shuffle7x7')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, bn=bn, ksize=7, stride=stride,
                                             block_mode='ShuffleNetV2', use_se=block_use_se, act_name=act_name, shuffle_method=shuffle_method))
                    elif blockIndex == 3:
                        # print('Xception')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, bn=bn, ksize=3, stride=stride,
                                             block_mode='ShuffleXception', use_se=block_use_se, act_name=act_name, shuffle_method=shuffle_method))
                    else:
                        raise NotImplementedError

                else:
                    archIndex += 1
                    self.features.append(ShuffleNasBlock(input_channel, output_channel, stride=stride, bn=bn,
                                                              max_channel_scale=self.candidate_scales[-1],
                                                              use_all_blocks=self.use_all_blocks,
                                                              use_se=block_use_se, act_name=act_name))
                # update input_channel for next block
                input_channel = output_channel

        if fix_arch:
            self.features = nn.Sequential(*self.features)
        else:
            self.features = NasHybridSequential(self.features)

        if self.last_conv_after_pooling:
            self.conv_last = nn.Sequential(
                # GlobalAvgPool2d(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_channel,
                          self.stage_out_channels[-1], 1, 1, 0, bias=False),
                # nn.BatchNorm2d(self.stage_out_channels[-1]),
                Activation('hswish' if self.use_se else 'relu'),
            )
        else:
            if self.use_se:
                # ShuffleNetV2+ approach
                self.conv_last = nn.Sequential(
                    nn.Conv2d(input_channel, make_divisible(
                        self.stage_out_channels[-1] * 0.75), 1, 1, 0, bias=False),
                    bn(make_divisible(self.stage_out_channels[-1] * 0.75)),
                    nn.AdaptiveAvgPool2d(1),
                    SE(make_divisible(self.stage_out_channels[-1] * 0.75)),
                    nn.Conv2d(make_divisible(
                        self.stage_out_channels[-1] * 0.75), self.stage_out_channels[-1], 1, 1, 0, bias=False),
                    Activation('hswish' if self.use_se else 'relu'),
                )
            else:
                # Origin Oneshot NAS approach
                self.conv_last = nn.Sequential(
                   nn.Conv2d(input_channel,
                             self.stage_out_channels[-1], 1, 1, 0, bias=False),
                   bn(self.stage_out_channels[-1]),
                   Activation('hswish' if self.use_se else 'relu'),
                   nn.AvgPool2d(7)
                   # nn.AdaptiveAvgPool2d(1),
                )
        # self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.2 if self.use_se else 0.1)
        self.classifier = nn.Sequential(
            # nn.Conv2d(self.stage_out_channels[-1], n_class, 1,1,0,bias=False))
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
            Activation('hswish' if self.use_se else 'relu'),)
            # nn.Flatten())
        self._initialize_weights()

    # add flops constraint
    def random_block_choices(self, num_of_block_choices=4, select_predefined_block=False, timeout=1, full_channel_mask=None, flops_constraint=False):
        if select_predefined_block:
            block_choices = [0, 0, 3, 1, 1, 1, 0, 0,
                2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        else:
            block_number = sum(self.stage_repeats)
            if flops_constraint:
                # flosp constraint
                flops_l, flops_r, flops_step = 290, 360, 10
                bins = [[i, i+flops_step]
                    for i in range(flops_l, flops_r, flops_step)]
                idx = np.random.randint(len(bins))
                l, r = bins[idx]
                for i in range(timeout):
                    block_choices = []
                    for i in range(block_number):
                        block_choices.append(random.randint(
                            0, num_of_block_choices - 1))
                    if l*1e6 <= self.get_cand_flops(block_choices=block_choices, full_channel_mask=full_channel_mask) <= r*1e6:
                        return torch.Tensor(block_choices)
            else:
                block_choices = []
                for i in range(block_number):
                   block_choices.append(random.randint(
                       0, num_of_block_choices - 1))
        return torch.Tensor(block_choices) 
        # return block_choices  # if use tensor will split in dataparallel

    def get_cand_flops(self, input_shape=(1, 3, 224, 224), block_choices=None, full_channel_mask=None):
        
        list_conv = []
        def conv_hook(self, input, output):
            batch_size, input_channels, input_height, input_width = input[0].size(
            )
            output_channels, output_height, output_width = output[0].size()

            assert self.in_channels % self.groups == 0

            kernel_ops = self.kernel_size[0] * self.kernel_size[
                1] * (self.in_channels // self.groups)
            params = output_channels * kernel_ops
            flops = batch_size * params * output_height * output_width

            list_conv.append(flops)

        list_linear = []
        def linear_hook(self, input, output):
            batch_size = input[0].size(0) if input[0].dim() == 2 else 1

            weight_ops = self.weight.nelement()

            flops = batch_size * weight_ops
            list_linear.append(flops)

        def foo(model, block_choices):
            first_conv = model.first_conv
            for net in first_conv:
                if isinstance(net, torch.nn.Conv2d):
                    net.register_forward_hook(conv_hook)
            
            features = model.features
            for i,net in enumerate(features.blocks):
                block_choice = block_choices[i]
                #block = net.block_sn_3x3
                if block_choice == 0:
                    block = net.block_sn_3x3
                elif block_choice == 1:
                    block = net.block_sn_5x5
                elif block_choice == 2:
                    block = net.block_sn_7x7
                elif block_choice == 3:
                    block = net.block_sx_3x3
                
                branch_main = block.branch_main
                if block.stride == 2:
                    branch_proj = block.branch_proj
                    for branch in branch_proj:
                        if isinstance(branch, torch.nn.Conv2d):
                            branch.register_forward_hook(conv_hook)
                        if isinstance(branch, torch.nn.Linear):
                            branch.register_forward_hook(linear_hook)
                for branch in branch_main.blocks:
                    if isinstance(branch, torch.nn.Conv2d):
                        branch.register_forward_hook(conv_hook)
                    if isinstance(branch, torch.nn.Linear):
                        branch.register_forward_hook(linear_hook)
                    if isinstance(branch, SE):
                        for se in branch.channel_attention:
                            if isinstance(se, torch.nn.Conv2d):
                                se.register_forward_hook(conv_hook)
                            if isinstance(se, torch.nn.Linear):
                                se.register_forward_hook(linear_hook)
            
            conv_last = model.conv_last
            for net in conv_last:
                if isinstance(net, torch.nn.Conv2d):
                    net.register_forward_hook(conv_hook)
            
            classifier = model.classifier
            for net in classifier:
                if isinstance(net, torch.nn.Linear):
                    net.register_forward_hook(linear_hook)
            return
        

        model = self
        input = torch.randn(1,3,224,224)
        foo(model, block_choices)
        out = model(input, torch.Tensor(block_choices), full_channel_mask)
        
        total_flops = sum(sum(i) for i in [list_conv, list_linear])
        print(total_flops / 1e6, 'M')
        return total_flops

    def random_channel_mask(self, select_all_channels=False, mode='sparse', epoch_after_cs=maxsize,
                            ignore_first_two_cs=False):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels) - 3

        # From [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] to [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], warm-up stages are
        # not just 1 epoch, but 2, 3, 4, 5 accordingly.
        
        epoch_delay_early = {0: 0,  # 8
                             1: 1, 2: 1,  # 7
                             3: 2, 4: 2, 5: 2,  # 6
                             6: 3, 7: 3, 8: 3, 9: 3,  # 5
                             10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
                             15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
                             21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 26:6 ,27: 6, 28: 6,
                             29: 6, 30: 6, 31: 6, 32: 6, 33: 6, 34: 6, 35: 6, 36: 7,
                           }
        epoch_delay_late = {0: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            4: 4, 5: 4,  # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
                            6: 5, 7: 5, 8: 5,  # warm up epoch: 3 ...
                            9: 6, 10: 6, 11: 6, 12: 6,  # warm up epoch: 4 ...
                            13: 7, 14: 7, 15: 7, 16: 7, 17: 7,  # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
                            18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, after 17, use all scales
        
        if 0 <= epoch_after_cs <=23 and self.stage_out_channels[2] >= 64:
            delayed_epoch_after_cs = epoch_delay_late[epoch_after_cs]
        elif 0<=epoch_after_cs <=36  and self.stage_out_channels[2] < 64:
            delayed_epoch_after_cs = epoch_delay_early[epoch_after_cs]
        else:
            delayed_epoch_after_cs = epoch_after_cs

        if ignore_first_two_cs:
            min_scale_id = 2
        else:
            min_scale_id = 0

        channel_mask = []
        channel_choices = []
        global_max_length = int(self.stage_out_channels[-1] // 2 * self.candidate_scales[-1])
        for i in range(len(self.stage_out_channels[2:-1])):
            local_max_length = int(self.stage_out_channels[i] // 2 * self.candidate_scales[-1])
            local_min_length = int(self.stage_out_channels[i] // 2 * self.candidate_scales[0])
            for _ in range(self.stage_repeats[i]):
                if select_all_channels:
                    local_mask = [1] * global_max_length
                else:
                    local_mask = [0] * global_max_length
                    # TODO: shouldn't random between min and max. But select candidate scales and return it.
                    if mode == 'dense':
                        random_select_channel = random.randint(local_min_length, local_max_length)
                        # In dense mode, channel_choices is # channel
                        channel_choices.append(random_select_channel)
                    elif mode == 'sparse':
                        # this is for channel selection warm up: channel choice ~ (8, 9) -> (7, 9) -> ... -> (0, 9)
                        channel_scale_start = max(min_scale_id, len(self.candidate_scales) - delayed_epoch_after_cs - 2)
                        channel_choice = random.randint(channel_scale_start, len(self.candidate_scales) - 1)
                        random_select_channel = int(self.stage_out_channels[i] // 2 *
                                                    self.candidate_scales[channel_choice])
                        # In sparse mode, channel_choices is the indices of candidate_scales
                        channel_choices.append(channel_choice)
                        random_select_channel = make_divisible(random_select_channel)
                    else:
                        raise ValueError("Unrecognized mode: {}".format(mode))
                    for j in range(random_select_channel):
                        local_mask[j] = 1
                channel_mask.append(local_mask)
        return torch.FloatTensor(channel_mask), channel_choices
        #return channel_mask, channel_choices #for dataparallel

    def forward(self, x, full_arch, full_scale_mask):
        x = self.first_conv(x)
        if full_arch is None:
            x = self.features(x)
        else:
            x = self.features(x, full_arch, full_scale_mask)
        x = self.conv_last(x)
        # x = self.globalpool(x)
        
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, NasBatchNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def count_flops(self):
        return None

class ShuffleNetV2_OneShotFix(ShuffleNetV2_OneShot):
    # Unlike its parent class, fix-arch model does not have the control of "use_all_blocks" and "bn"(for NasBN).
    # It should use the default False and nn.BatchNorm correspondingly.
    def __init__(self, input_size=224, n_class=1000, architecture=None, channels_scales=None,
                 use_se=False, last_conv_after_pooling=False, shuffle_method=ShuffleChannels, stage_out_channels=None, candidate_scales=None):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNetV2_OneShotFix, self).__init__(input_size=input_size, n_class=n_class,
                                                   architecture=architecture, channels_scales=channels_scales, candidate_scales=candidate_scales,
                                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling, shuffle_method=shuffle_method)

    def forward(self, x, *args, **kwargs):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        # x = self.globalpool(x)
        
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

def get_shufflenas_oneshot(architecture=None, scale_ids=None, use_all_blocks=False, n_class=1000,
                           use_se=False, last_conv_after_pooling=False, shuffle_by_conv=False, channels_layout='OneShot'):
    
    if channels_layout == 'OneShot':
        stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    elif channels_layout == 'ShuffleNetV2+':
        stage_out_channels = [-1, 16, 48, 128, 256, 512, 1024]
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    else:
        stage_out_channels = []
        raise ValueError("Unrecognized channels_layout: {}. "
                         "Please choose from ['ShuffleNetV2', 'OneShot']".format(channels_layout))

    if architecture is None and scale_ids is None:
        # Nothing about architecture is specified, do random block selection and channel selection.
        net = ShuffleNetV2_OneShot(n_class=n_class, use_all_blocks=use_all_blocks,bn = NasBatchNorm,
                                use_se=use_se, last_conv_after_pooling=last_conv_after_pooling,
                                stage_out_channels=stage_out_channels, candidate_scales=candidate_scales)
    elif architecture is not None and scale_ids is not None:
        # Create the specified structure
        if use_all_blocks:
            raise ValueError("For fixed structure, use_all_blocks should not be allowed.")
        shuffle_method = ShuffleChannelsConv if shuffle_by_conv else ShuffleChannels
        scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        channels_scales = []
        for i in range(len(scale_ids)):
            # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
            channels_scales.append(scale_list[scale_ids[i]])
        net = ShuffleNetV2_OneShotFix(architecture=architecture, channels_scales=channels_scales,
                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling,
                                   stage_out_channels=stage_out_channels, candidate_scales=candidate_scales, shuffle_method=shuffle_method)
    else:
        raise ValueError("architecture and scale_ids should both be None for supernet "
                         "or both not None for fixed structure model.")
    return net

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

def get_flops(model):
    input = torch.randn(1,3,224,224)
    # flops, params = profile(model, input=(input, ), custom_ops={model: count_flops})
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: ', flops, 'params: ', params)

FIX_ARCH = False 
LAST_CONV_AFTER_POOLING = True
USE_SE = True
SHUFFLE_BY_CONV = False
CHANNELS_LAYOUT = 'OneShot'

def main():
    # from calculate_flops import get_flops

    if FIX_ARCH:
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                     use_se=USE_SE, last_conv_after_pooling=LAST_CONV_AFTER_POOLING)
    else:
        net = get_shufflenas_oneshot(use_se=USE_SE, last_conv_after_pooling=LAST_CONV_AFTER_POOLING)

    """ Test customized initialization """
    net._initialize_weights()
    print(np.sum(np.prod(v.size()) for name, v in net.named_parameters() if "auxiliary" not in name)/1e6) 
    test_data = torch.rand(1, 3, 224, 224)
    ### Test the blocks constriant of flosp ###
    full_channel_mask, _ = net.random_channel_mask(select_all_channels=False)
    block_choices = net.random_block_choices(select_predefined_block=False, full_channel_mask=full_channel_mask, flops_constraint=True)

    a = torch.zeros((20, 1024))
    test_outputs = net(test_data, block_choices, a)
    #test_outputs = net(test_data, block_choices, full_channel_mask)
    print(test_outputs.shape)
    #test_outputs = net(test_data)
    # print(summary(net, [(3,224,224), (10,1),(10,1)]))

    """ Test ShuffleNasOneShot """
    for epoch in range(1):
        if FIX_ARCH:
            test_outputs = net(test_data)
        else:
            block_choices = net.random_block_choices(select_predefined_block=False)
            full_channel_mask, channel_choices = net.random_channel_mask(
                select_all_channels=False)
            test_outputs = net(test_data, block_choices, full_channel_mask)
            print(test_outputs.size())
        
    """ Test generating random channels """
    epoch_start_cs = 60
    use_all_channels = True if epoch_start_cs != -1 else False

    for epoch in range(120):
        if epoch == epoch_start_cs:
            use_all_channels = False
        for batch in range(1):
            full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=use_all_channels,
                                                                            epoch_after_cs=epoch - epoch_start_cs,
                                                                            ignore_first_two_cs=True)
            print("Epoch {}: {}".format(epoch, channel_choices))

    # TODO: remove debug code
    stage_repeats = net.stage_repeats
    stage_out_channels = net.stage_out_channels
    candidate_scales = net.candidate_scales
    channel_scales = []
    for c in range(len(channel_choices)):
        # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        channel_scales.append(
            candidate_scales[channel_choices[c]])
    channels = [stage_out_channels[0]] * stage_repeats[0] + \
                [stage_out_channels[1]] * stage_repeats[1] + \
                [stage_out_channels[2]] * stage_repeats[2] + \
                [stage_out_channels[3]] * stage_repeats[3]
    channels = [make_divisible(channel / 2 * channel_scales[j]) for (j, channel)
                in enumerate(channels)]
    print("Train batch channel choices: {}".format(
        channel_choices))
    print("Train batch channels: {}".format(channels))
    # TODO: end of debug code

    # get_flops(net)

if __name__ == '__main__':
    main()
