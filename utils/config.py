'''
@Description: Config class for config
@Author: xieydd
@Date: 2019-08-13 14:01:03
@LastEditTime: 2019-09-30 18:13:12
@LastEditors: Please set LastEditors
'''
import argparse
import os
import torch
from functools import partial

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

'''
@description: for mutil gpus train in pytorch
@param string "0,1,2,3" 
@return: gpu list [0, 1, 2, 3]
'''
def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

'''
@description: Base config class include a print params function and as_markdown function
@param {type} 
@return: 
'''
class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)
        return text


'''
@description: For SPOS Config 
'''
class SPOSConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('-a', '--arch', metavar='ARCH',default='Darts',
                    help='model architecture (default: Darts)')
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST / ImageNet')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
        parser.add_argument('--val_portion', type=float, default=0.5, help='portion of validate data')
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
        parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
        parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
        parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
        parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext and ShuffleNas. default is false.')
        parser.add_argument('--epoch-start-cs', type=int, default=224,
                        help='Epoch id for starting Channel selection.')
        parser.add_argument('--use-all-blocks', action='store_true',
                        help='whether to use all the choice blocks.')
        parser.add_argument('--use-all-channels', action='store_true',
                            help='whether to use all the channels.')
        parser.add_argument('--last-conv-after-pooling', action='store_true',
                            help='Whether to follow MobileNet V3 last conv after pooling style.')
        parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
        parser.add_argument('--cs-warm-up', action='store_true',
                        help='Whether to do warm up for Channel Selection so that gradually selects '
                             'larger range of channels')
        parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
        parser.add_argument('--reduced-dataset-scale', type=int, default=1,
                        help='How many times the dataset would be reduced, so that in each epoch '
                             'only num_batches / reduced_dataset_scale batches will be trained.')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)
