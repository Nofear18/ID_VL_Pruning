import os
import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_nlvr import blip_nlvr,blip_nlvr_Pruned
from ID_pruner_nlvr import *
import utils
from utils import cosine_lr_schedule, warmup_lr_schedule
from data import create_dataset, create_sampler, create_loader
from torch.utils.tensorboard import SummaryWriter

def train(model, args,data_loader,global_step, optimizer, epoch, device, config):
    # train
    model.train()
    if args.max_steps > 0:
        t_total = args.max_steps
        config['max_epoch'] = args.max_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(data_loader) // args.gradient_accumulation_steps * config['max_epoch']

    PLATON = Pruner(model, args=args, total_step=t_total, use_no_mask=False, pruner_name=args.pruner_name)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 10

    for i, (image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)

        loss = model(images, text, targets=targets, train=True)

        optimizer.zero_grad()
        loss.backward()

        if (global_step % len(data_loader) != 0):
            optimizer.step()
        useID = args.useID

        ID_BLIP_nlvr = []#should be computed
        ID = torch.tensor(ID_BLIP_nlvr)
        PLATON.update_and_pruning(args, model, len(data_loader), global_step, ID, useID)
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if (global_step % len(data_loader) == 0) or (global_step == 1):
            EPOCH = epoch
            modelPath =  args.result_dir + '/models/'
            Path(modelPath).mkdir(parents=True, exist_ok=True)
            PATH = modelPath + 'epoch_{}globalstep_{}.pth'.format(
                epoch, global_step)

            LOSS = loss

            writer.add_scalar('Train/loss', LOSS.item(), t_total)

            torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                'lr_schedule': optimizer.param_groups[0]["lr"],
            }, PATH)
            print(global_step)
            print("successfully saved")

        if (global_step % len(data_loader) == 0):
            break
        global_step = global_step + 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for image0, image1, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)

        prediction = model(images, text, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    datasets = create_dataset('nlvr', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    batch_size = [config['batch_size_train'], config['batch_size_test'], config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets, samplers, batch_size=batch_size,
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    #### Model ####
    print("Creating model")
    if args.evaluate and args.pruned:
        model = blip_nlvr_Pruned(pretrained=args.pruned, image_size=config['image_size'], vit=config['vit'],
                                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    else:
        model = blip_nlvr(pretrained=config['pretrained'], image_size=config['image_size'],
                      vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            global_step = epoch * len(train_loader) + 1
            train_stats = train(model, args,train_loader,global_step, optimizer, epoch, device, config)

        val_stats = evaluate(model, val_loader, device, config)
        test_stats = evaluate(model, test_loader, device, config)

        if utils.is_main_process():
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             }
                with open(os.path.join(args.result_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             }

                if float(val_stats['acc']) > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = float(val_stats['acc'])
                    best_epoch = epoch

                with open(os.path.join(args.result_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        if args.evaluate:
            break

        dist.barrier()

    if utils.is_main_process():
        with open(os.path.join(args.result_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # adaptive pruning
    parser.add_argument('--pruner_name', default='Taylor', type=str,
                        help="[PLATON, Taylor,Magnitude]")
    parser.add_argument('--beta1', default=0.85, type=float, help="beta1 for PLATON.")
    parser.add_argument('--deltaT', default=10, type=int, help="The legnth of local window to reweight EMA.")
    parser.add_argument('--beta2', default=0.95, type=float, help="beta2 for PLATON")
    # pruning schedule
    parser.add_argument('--warmup_steps', default=4318, type=int, help="Warmup steps.")
    parser.add_argument('--initial_threshold', default=1., type=float, help="Initial threshold.")
    parser.add_argument('--final_threshold', default=0.20, type=float, help="Final threshold.")
    parser.add_argument('--initial_warmup', default=0, type=int, help="Initial Warmup for Pruner.")
    parser.add_argument('--final_warmup', default=5, type=int, help="Final Warmup for Pruner.")

    parser.add_argument("--n_gpu", default=2, type=int, help="number of gpu.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument('--useID', action='store_true')
    parser.add_argument('--pruned', default='', type=str, help="Path of the pruned model")

    parser.add_argument('--config', default='configs/nlvr.yaml')
    parser.add_argument('--output_dir', default='BLIP_nlvr')
    parser.add_argument('--model_dir', default='Test')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')  # '"cuda:1"'
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, args.model_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    # DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    writer = SummaryWriter(log_dir=os.path.join(
        args.result_dir, 'runs', '1'))
    print('useID{}'.format(args.useID))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)