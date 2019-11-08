'''
@Description: Train Search SuperNet Single Path One Shot Nas
@Author: xieydd
@Date: 2019-10-05 15:37:47
@LastEditTime: 2019-10-11 18:22:59
@LastEditors: Please set LastEditors
'''
import os
import sys
import time
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils.config import SPOSConfig
from model.network import get_shufflenas_oneshot
from model.blocks import CrossEntropyLabelSmooth

from utils.imagenet_dataloader import get_imagenet_iter_torch
from utils import utils

config = SPOSConfig()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(
    config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    start = time.time()
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(config.local_rank % len(config.gpus))
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    config.world_size = torch.distributed.get_world_size()
    config.total_batch = config.world_size * config.batch_size

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    CLASSES = 1000
    num_training_samples = 1281167
    num_batches = num_training_samples * config.train_portion // config.batch_size
    model_name = config.arch

    if config.epoch_start_cs != -1:
        config.use_all_channels = True

       # Model
    if model_name == 'ShuffleNas_fixArch':
        architecture = [0, 0, 3, 1, 1, 1, 0, 0,
                        2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_ids = [6, 5, 3, 5, 2, 6, 3, 4,
                     2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        # we rescale the channels uniformly to adjust the FLOPs.
        model = get_shufflenas_oneshot(architecture, scale_ids, use_se=config.use_se, n_class=CLASSES,
                                       last_conv_after_pooling=config.last_conv_after_pooling,
                                       channels_layout=config.channels_layout)
    elif model_name == 'ShuffleNas':
        model = get_shufflenas_oneshot(use_all_blocks=config.use_all_blocks, use_se=config.use_se, n_class=CLASSES,
                                       last_conv_after_pooling=config.last_conv_after_pooling,
                                       channels_layout=config.channels_layout)
    else:
        raise NotImplementedError

    model = model.to(device)
    model.apply(utils.weights_init)
    # model = DDP(model, delay_allreduce=True)
    # For solve the custome loss can`t use model.parameters() in apex warpped model via https://github.com/NVIDIA/apex/issues/457 and https://github.com/NVIDIA/apex/issues/107
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.local_rank], output_device=config.local_rank)
    if model_name == 'ShuffleNas_fixArch':
        logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    else:
        logger.info("Train Supernet")
    # Loss
    if config.label_smoothing:
        criterion = CrossEntropyLabelSmooth(CLASSES, config.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    weight = model.parameters()
    # Optimizer
    w_optimizer = torch.optim.SGD(
        weight,
        0.01,
        momentum=config.w_momentum,
        weight_decay=config.w_weight_decay)

    w_schedule = utils.Schedule(w_optimizer)

    train_data = get_imagenet_iter_torch(
        type='train',
        # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
        # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/`
        image_dir=config.data_path+config.dataset.lower(),
        batch_size=config.batch_size,
        num_threads=config.workers,
        world_size=config.world_size,
        local_rank=config.local_rank,
        crop=224, device_id=config.local_rank, num_gpus=config.gpus, portion=config.train_portion
    )

    valid_data = get_imagenet_iter_torch(
        type='train',
        # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
        # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/`
        image_dir=config.data_path+"/"+config.dataset.lower(),
        batch_size=config.batch_size,
        num_threads=config.workers,
        world_size=config.world_size,
        local_rank=config.local_rank,
        crop=224, device_id=config.local_rank, num_gpus=config.gpus, portion=config.val_portion
    )

    config.start_epoch = 0
    if config.resume:
        lastest_model, iters = utils.get_lastest_model(config.path)
        if lastest_model is not None:
            config.start_epoch = iters
            w_optimizer = utils.load_checkpoint_spos(
                model, lastest_model)
            w_schedule.optimizer = w_optimizer
            for i in range(iters):
                w_scheduler.step(i)

    best_top1 = 0.
    for epoch in range(config.start_epoch, config.epochs):
        if epoch < config.warmup_epochs:
            lr = w_schedule.schedule.update_schedule_linear_target(
                config.warmup_epochs, epoch, config.w_lr num_batches)
        else:
            w_scheduler = w_schedule.get_schedule_cosine(
                config.w_lr_min, config.epochs)
            w_scheduler.step(epoch)
            lr = w_scheduler.get_lr()[0]

        if epoch > config.epoch_start_cs:
            config.use_all_channels = False
        # training
        optimizer = w_schedule.optimizer
        train_top1, train_loss = train(
            train_data, valid_data, model, criterion, w_optimizer, lr, epoch, writer, model_name)
        logger.info('Train top1 %f', train_top1)

        # validation
        top1 = 0
        if epoch % 10 == 0:
            top1, loss = infer(valid_data, model, epoch,
                               criterion, writer, model_name)
            logger.info('valid top1 %f', top1)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        if epoch % config.save_iterval == 0:
            utils.save_checkpoint_spos({
                'state_dict': model.state_dict(),
                'w_optimizer': w_schedule.optimizer,
            }, epoch, config.path, is_best)
        print("")

    utils.time(time.time() - start)
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_queue, valid_queue, model, criterion, optimizer, lr, epoch, writer, model_name):
    batch_time = utils.AverageMeters('Time', ':6.3f')
    data_time = utils.AverageMeters('Data', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')

    progress = utils.ProgressMeter(len(train_queue), batch_time, data_time, losses, top1,
                                   top5, prefix="Epoch: [{}]".format(epoch))
    cur_step = epoch*len(train_queue)

    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    end = time.time()
    if config.mixup:
        beta_distribution = torch.distributions.beta.Beta(
            config.mixup_alpha, config.mixup_alpha)

    for step, (input, target) in enumerate(train_queue):

        if epoch < config.warmup_epochs:
            lr_step = lr * (step + 1) * len(config.gpus)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_step
            if step % config.print_freq == 0:
                print('lr: %e' % lr_step)
        # measure data loading time
        data_time.update(time.time() - end)

        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        target = Variable(target, requires_grad=False).cuda()
        optimizer.zero_grad()

        if config.mixup:
            # Mixup images.
            lambda_ = beta_distribution.sample([]).item()
            index = torch.randperm(input.size(0)).cuda()
            mixed_images = lambda_ * input + (1 - lambda_) * input[index, :]

            # Mixup loss.
            scores = model(mixed_images)
            loss = (lambda_ * criterion(scores, target)
                    + (1 - lambda_) * criterion(scores, target[index]))

        if model_name == "ShuffleNas":
            block_choices = model.module.random_block_choices(
                select_predefined_block=False)
            if config.cs_warm_up:
                full_channel_mask, _ = model.module.random_channel_mask(select_all_channels=config.use_all_channels,
                                                                        epoch_after_cs=epoch - config.epoch_start_cs, ignore_first_two_cs=config.ignore_first_two_cs)
            else:
                full_channel_mask, _ = model.module.random_channel_mask(
                    select_all_channels=config.use_all_channels, ignore_first_two_cs=config.ignore_first_two_cs)
            logits = model(input, block_choices, full_channel_mask)
        else:
            logits = model(input, None, None)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), config.w_grad_clip)
        # if model_name == "ShuffleNas":
        #     weight = model.parameters()
        #     optimizer = torch.optim.SGD(
        #         weight,
        #         lr,
        #         momentum=config.w_momentum,
        #         weight_decay=config.w_weight_decay)
        #     optimizer.step()
        # else:
        #     optimizer.step()
        optimizer.step()
        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
        acc1 = reduce_tensor(acc1, world_size=config.world_size)
        acc5 = reduce_tensor(acc5, world_size=config.world_size)

        losses.update(to_python_float(reduced_loss), n)
        top1.update(to_python_float(acc1), n)
        top5.update(to_python_float(acc5), n)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0 or step == len(train_queue)-1:
            logger.info('train step:%03d %03d  loss:%e top1:%05f top5:%05f', step, len(
                train_queue), losses.avg, top1.avg, top5.avg)
            progress.print(step)
        writer.add_scalar('train/loss', losses.avg, cur_step)
        writer.add_scalar('train/top1', top1.avg, cur_step)
        writer.add_scalar('train/top5', top5.avg, cur_step)

    return top1.avg, losses.avg


def infer(valid_queue, model, epoch, criterion, writer, model_name):
    batch_time = utils.AverageMeters('Time', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')
    model.eval()

    progress = utils.ProgressMeter(len(valid_queue), batch_time, losses, top1, top5,
                                   prefix='Test: ')
    cur_step = epoch*len(valid_queue)

    end = time.time()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            # input = input.cuda()
            # target = target.cuda(non_blocking=True)
            input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(async=True)
            target = Variable(target, volatile=True).cuda()
            if model_name == "ShuffleNas":
                block_choices = model.module.random_block_choices(
                    select_predefined_block=False)
                if config.cs_warm_up:
                    full_channel_mask, _ = model.module.random_channel_mask(select_all_channels=True,
                                                                            epoch_after_cs=epoch - config.epoch_start_cs, ignore_first_two_cs=config.ignore_first_two_cs)
                else:
                    full_channel_mask, _ = model.module.random_channel_mask(
                        select_all_channels=True,  ignore_first_two_cs=config.ignore_first_two_cs)
                logits = model(input, block_choices, full_channel_mask)
            else:
                logits = model(input, None, None)
            loss = criterion(logits, target)
            acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            reduced_loss = reduce_tensor(
                loss.data, world_size=config.world_size)
            acc1 = reduce_tensor(acc1, world_size=config.world_size)
            acc5 = reduce_tensor(acc5, world_size=config.world_size)
            losses.update(to_python_float(reduced_loss), n)
            top1.update(to_python_float(acc1), n)
            top5.update(to_python_float(acc5), n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % config.print_freq == 0:
                progress.print(step)
                logger.info('valid %03d %e %f %f', step,
                            losses.avg, top1.avg, top5.avg)

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)
    return top1.avg, losses.avg


def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def adjust_learning_rate(optimizer, warmup_epoch, epoch, args, nums_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (args.lr_decay ** (epoch // nums_batches))
    else:
        if epoch <= warmup_epoch:
            lr = args.lr * (args.lr_decay ** (epoch // nums_batches))
        else:
            lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.w_lr_min)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
