import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import argparse
import sys
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils.prune
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.ao import sparsity
from torch.utils.data import DataLoader

from models.blip import blip_decoder,blip_decoder_Pruned
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval


import glob
import logging

import random
import timeit

import numpy as np
from ID_pruner import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange



def train(model, args, data_loader, global_step, optimizer, epoch, device):
    # train
    model.train()

    if args.max_steps > 0:
        t_total = args.max_steps
        config['max_epoch'] = args.max_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(data_loader) // args.gradient_accumulation_steps * config['max_epoch']


    PLATON = Pruner(model, args=args, total_step=t_total, use_no_mask=False, pruner_name=args.pruner_name)


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print(header)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        loss = model(image, caption)
        optimizer.zero_grad()
        loss.backward()

        if (global_step % len(data_loader) != 0):
            optimizer.step()
        useID = args.useID
        #ID computed by TwoNN
        ID_blip_coco = [39.2525453585, 26.2860314495, 56.858500081500004, 80.1758923525, 46.2581305575, 27.431526575499998,
                  58.88023918200001, 57.9350088635, 53.672930027999996, 40.877771837, 88.37791575099999, 81.8891860435,
                  72.9864716375, 64.69155517850001, 109.2069528345, 97.047251945, 91.95283622349999, 77.0256082645,
                  106.68032645449998, 141.53732612250002, 106.772897611, 68.7427169435, 86.704447321, 65.67260548799999,
                  89.469670297, 69.7204040395, 78.1054637245, 30.434839641499984, 91.36768140750002, 58.20166747649999,
                  70.695042714, 19.3650993175, 87.64788148799998, 58.07942919300001, 70.0856539205, 58.683572293,
                  78.66800641500001, 52.214217995, 74.97459880400001, 64.35572059799999, 84.9143167015,
                  41.66979977650001, 78.6355654555, 39.420815939, 81.27035343550001, 40.490194347999996, 59.9451197,
                  32.030728378, 14.105554355999999, 11.547037749000001, 21.290597912000003, 20.6240448395,
                  20.869356689999996, 65.7867036575, 80.642361986, 15.299692193499999, 12.853699612999998, 3.783597073,
                  10.049439903499998, 10.587040969499999, 19.62432913, 19.605800026, 21.16074337, 66.342837129,
                  71.56374188699999, 16.272560202, 19.0308684615, 5.4794352029999995, 13.681458621999997,
                  12.234400914000002, 19.05776608, 21.417493471, 19.186245793, 76.7624444585, 68.87110796300001,
                  13.8109685315, 15.837269498499998, 1.8982450184999997, 13.646299074500002, 12.7617679195,
                  19.731017737000002, 28.3450230395, 17.345281847000003, 76.8490473205, 77.88905990700002,
                  22.9515276295, 14.998424784000003, 2.2580321100000003, 17.099025393500003, 16.6374507355,
                  19.264338590499996, 61.010588502000004, 18.522133783999998, 78.61743028449999, 79.9224954025,
                  42.171300439999996, 16.870131960499997, 2.3452206374999998, 18.222086760000003, 16.9889144215,
                  18.795345242, 34.238710419, 18.115870945, 79.255367338, 72.68314991, 14.849818867500002, 16.542755932,
                  2.5533363245, 18.055293564499998, 17.3016755885, 18.997541575, 47.733564533999996, 13.622358453500002,
                  79.64567560249999, 76.8646484805, 41.2187293665, 17.910362366, 2.277890411, 19.4179970235,
                  15.335755064000002, 23.4514201415, 40.789648255500005, 16.256408872999998, 81.17477228499999,
                  79.5234647535, 29.592402433500002, 17.048154381999996, 2.306545864, 19.214582003500002, 16.95310157,
                  22.492424913, 40.899307455, 16.761272939500003, 80.165535647, 79.2239572895, 27.347409656000004,
                  17.573595676, 5.571349004500001, 22.352257993000002, 17.1987938165, 21.369393836, 39.296991567999996,
                  20.615685399500002, 78.194997336, 75.3874604965, 21.345088765999996, 19.6440573975, 9.2058756515,
                  18.349747171500002, 18.439094996499996, 22.607490932000005, 27.4353717615, 20.948007669499994,
                  78.8095940365, 75.015445368, 24.033069358000002, 20.5939448875, 13.532389413500002,
                  18.350829726500002, 17.696479891, 23.812569949999997, 28.0342546115, 20.2689085205, 79.15692295999999,
                  69.15721603350002, 22.8789594485, 18.313451692, 3.809255495]

        ID = torch.tensor(ID_blip_coco)
        PLATON.update_and_pruning( args,model, len(data_loader), global_step, ID, useID)
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        #Save model
        if (global_step % len(data_loader) == 0):
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
            print("successfully saved")

        if (global_step % len(data_loader) == 0):
            break
        global_step = global_step + 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)

        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


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
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks,
                                  global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size']] * 3, num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    #### Model ####
    print("Creating model")
    
    """
    If you want load pruned model, use blip_decoder_Pruned
    """
    if args.evaluate and args.pruned:
        model = blip_decoder_Pruned(pretrained=args.pruned, image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])
    else:
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])

    model = model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            lr = cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            print("-------------------train-------------------")
            global_step = epoch * len(train_loader) + 1
            train_stats = train(model, args, train_loader, global_step, optimizer, epoch, device)

        print("-------------------val-----------------------")
        val_result = evaluate(model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d' % epoch, remove_duplicate='image_id')
        print("-------------------test----------------------")
        test_result = evaluate(model_without_ddp, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d' % epoch,
                                       remove_duplicate='image_id')

        if utils.is_main_process():
            coco_val = coco_caption_eval(config['coco_gt_root'], val_result_file, 'val')
            coco_test = coco_caption_eval(config['coco_gt_root'], test_result_file, 'test')

            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             }
                with open(os.path.join(args.result_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                # torch.save(save_obj,os.path.join(args.output_dir,'checkpoint_epoch%d.pth'%epoch ))
                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint.pth'))

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.result_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                writer.add_scalar('Val cider', coco_val.eval['CIDEr'], epoch)
                writer.add_scalar('Val cider', lr, epoch)
        if args.evaluate:
            break
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # adaptive pruning
    parser.add_argument('--pruner_name', default='PLATON', type=str,
                        help="[PLATON, Taylor,Magnitude]")
    parser.add_argument('--beta1', default=0.85, type=float, help="beta1 for PLATON.")
    parser.add_argument('--deltaT', default=10, type=int, help="The legnth of local window to reweight EMA.")
    parser.add_argument('--beta2', default=0.95, type=float, help="beta2 for PLATON")
    # pruning schedule
    parser.add_argument('--warmup_steps', default=14168, type=int, help="Warmup steps.")
    parser.add_argument('--initial_threshold', default=1., type=float, help="Initial threshold.")
    parser.add_argument('--final_threshold', default=0.20, type=float, help="Final threshold.")
    parser.add_argument('--initial_warmup', default=1, type=int, help="Initial Warmup for Pruner.")
    parser.add_argument('--final_warmup', default=1, type=int, help="Final Warmup for Pruner.")
    parser.add_argument('--useID', action='store_true')
    parser.add_argument('--pruned', default='',type=str, help="Path of the pruned model")

    parser.add_argument("--n_gpu", default=2, type=int, help="number of gpu.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument('--config', default='configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/BLIP_coco')
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
    writer = SummaryWriter(log_dir=os.path.join(
        args.result_dir, 'runs', '1'))
    print('useID:{}'.format(args.useID))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)
    writer.close()