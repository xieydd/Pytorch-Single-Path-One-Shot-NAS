'''
@Description: ShuffleNet_OneShot
@Author: your name
@Date: 2019-09-26 19:44:29
@LastEditTime: 2019-10-11 22:45:14
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
from blocks import ShuffleNetBlock, Activation, ShuffleNasBlock, SE, NasHybridSequential 
import random
from sys import maxsize

class ShuffleNetV2_OneShot(nn.Module):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channels_scales=None,
                use_all_blocks=False, use_se=False, last_conv_after_pooling=False):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.use_all_blocks = use_all_blocks
        self.use_se = use_se
        self.last_conv_after_pooling = last_conv_after_pooling

        if architecture is None and channels_scales is None:
            fix_arch = False
        elif architecture is not None and channels_scales is not None:
            fix_arch = True
            assert len(architecture) == len(channels_scales)
        else:
            raise ValueError("architecture and scale_ids should be both None or not None.")
        self.fix_arch = fix_arch
        assert len(self.stage_repeats) == len(self.stage_out_channels) - 3 
        
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
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
                    inp, outp, stride = input_channel, output_channel, 1

                if fix_arch:
                    blockIndex = architecture[archIndex]
                    base_mid_channels = outp // 2
                    mid_channels = int(base_mid_channels * channels_scales[archIndex])
                    archIndex += 1
                    if blockIndex == 0:
                        print('Shuffle3x3')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride, block_mode='ShuffleNetV2', use_se=block_use_se, act_name=act_name))
                    elif blockIndex == 1:
                        print('Shuffle5x5')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride, block_mode='ShuffleNetV2', use_se=block_use_se, act_name=act_name))
                    elif blockIndex == 2:
                        print('Shuffle7x7')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride, block_mode='ShuffleNetV2', use_se=block_use_se, act_name=act_name))
                    elif blockIndex == 3:
                        print('Xception')
                        self.features.append(ShuffleNetBlock(inp, outp, mid_channels=mid_channels, ksize=3 ,stride=stride, block_mode='ShuffleXception', use_se=block_use_se, act_name=act_name))
                    else:
                        raise NotImplementedError
                   
                else:
                    archIndex += 1
                    self.features.append(ShuffleNasBlock(input_channel, output_channel, stride=stride,
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
                nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.stage_out_channels[-1]),
                Activation('hswish' if self.use_se else 'relu'),
            )
        else:
            if self.use_se:
                # ShuffleNetV2+ approach
                self.conv_last = nn.Sequential(
                    nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                    nn.BatchNorm2d(self.stage_out_channels[-1]),
                    nn.AdaptiveAvgPool2d(1),
                    SE(self.stage_out_channels[-1]),
                    nn.Conv2d(self.stage_out_channels[-1], self.stage_out_channels[-1], 1, 1, 0, bias=False),
                    Activation('hswish' if self.use_se else 'relu'),
                )
            else:
                # Origin Oneshot NAS approach
                self.conv_last = nn.Sequential(
                   nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                   nn.BatchNorm2d(self.stage_out_channels[-1]),
                   Activation('hswish' if self.use_se else 'relu'),
                   nn.AdaptiveAvgPool2d(1), 
                )
        #self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.2 if self.use_se else 0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
            Activation('hswish' if self.use_se else 'relu'),)
        self._initialize_weights()

    def random_block_choices(self, num_of_block_choices=4, select_predefined_block=False, dtype='float32'):
        if select_predefined_block:
            block_choices = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        else:
            block_number = sum(self.stage_repeats)
            block_choices = []
            for i in range(block_number):
                block_choices.append(random.randint(0, num_of_block_choices - 1))
        return torch.FloatTensor(block_choices)
     
    def random_channel_mask(self, select_all_channels=False, mode='sparse', epoch_after_cs=maxsize,
                            ignore_first_two_cs=True):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels) - 3

        # From [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] to [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], warm-up stages are
        # not just 1 epoch, but 2, 3, 4, 5 accordingly.
        if 0 <= epoch_after_cs <= 23:
            epoch_delay = {0: 0,
                           1: 1,
                           2: 2,
                           3: 3,
                           4: 4,  5: 4,                               # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
                           6: 5,  7: 5,  8: 5,                        # warm up epoch: 3 ...
                           9: 6,  10: 6, 11: 6, 12: 6,                # warm up epoch: 4 ...
                           13: 7, 14: 7, 15: 7, 16: 7, 17: 7,         # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
                           18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, actually this stage is useless

            delayed_epoch_after_cs = epoch_delay[epoch_after_cs]
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
                    else:
                        raise ValueError("Unrecognized mode: {}".format(mode))
                    for j in range(random_select_channel):
                        local_mask[j] = 1
                channel_mask.append(local_mask)
        return torch.FloatTensor(channel_mask), channel_choice

    def forward(self, x, full_arch, full_scale_mask):
        x = self.first_conv(x)
        if len(full_arch) == 1:
            x = self.features(x)
        else:
            x = self.features(x, full_arch, full_scale_mask)
        x = self.conv_last(x)

        #x = self.globalpool(x)
        
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
            elif isinstance(m, nn.BatchNorm2d):
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

class ShuffleNetV2_OneShotFix(ShuffleNetV2_OneShot):
    # Unlike its parent class, fix-arch model does not have the control of "use_all_blocks" and "bn"(for NasBN).
    # It should use the default False and nn.BatchNorm correspondingly.
    def __init__(self, input_size=224, n_class=1000, architecture=None, channels_scales=None,
                 use_se=False, last_conv_after_pooling=False):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNetV2_OneShotFix, self).__init__(input_size=input_size, n_class=n_class,
                                                   architecture=architecture, channels_scales=channels_scales,
                                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling)

    def forward(self, x, *args, **kwargs):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        #x = self.globalpool(x)
        
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

def get_shufflenas_oneshot(architecture=None, scale_ids=None, use_all_blocks=False,
                           use_se=False, last_conv_after_pooling=False):
    if architecture is None and scale_ids is None:
        # Nothing about architecture is specified, do random block selection and channel selection.
        net = ShuffleNetV2_OneShot(use_all_blocks=use_all_blocks,
                                use_se=use_se, last_conv_after_pooling=last_conv_after_pooling)
    elif architecture is not None and scale_ids is not None:
        # Create the specified structure
        if use_all_blocks:
            raise ValueError("For fixed structure, use_all_blocks should not be allowed.")
        scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        channels_scales = []
        for i in range(len(scale_ids)):
            # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
            channels_scales.append(scale_list[scale_ids[i]])
        net = ShuffleNetV2_OneShotFix(architecture=architecture, channels_scales=channels_scales,
                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling)
    else:
        raise ValueError("architecture and scale_ids should both be None for supernet "
                         "or both not None for fixed structure model.")
    return net

FIX_ARCH = False
LAST_CONV_AFTER_POOLING = True
USE_SE = True

def main():
    #from calculate_flops import get_flops

    if FIX_ARCH:
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                     use_se=USE_SE, last_conv_after_pooling=LAST_CONV_AFTER_POOLING)
    else:
        net = get_shufflenas_oneshot(use_se=USE_SE, last_conv_after_pooling=LAST_CONV_AFTER_POOLING)

    """ Test customized initialization """
    net._initialize_weights()
    test_data = torch.rand(5, 3, 224, 224)
    block_choices = net.random_block_choices(select_predefined_block=False)
    full_channel_mask, _ = net.random_channel_mask(select_all_channels=False)
    test_outputs = net(test_data, block_choices, full_channel_mask)
    #test_outputs = net(test_data)
    print(test_outputs.size())
    


if __name__ == '__main__':
    main()