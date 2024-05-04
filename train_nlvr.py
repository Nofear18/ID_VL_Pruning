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

        ID_BLIP_nlvr = [40.07103360890651, 27.863494686174597, 42.649941862926354, 78.46094644575649, 47.45257537563246, 24.53381147610316, 55.190562849906144,
     55.79093090920212, 58.40327716141047, 34.40767464634344, 70.81377170917649, 76.1184961986902, 71.10511499530482,
     52.16719807752386, 77.85502196969055,  84.86938349764797, 66.29985108056195, 56.09862965106672, 73.13864124974438,
     70.16127100300555, 70.06308064285024, 48.328316483402105, 61.376687393160715, 32.08788326508452, 63.65910382384377, 49.29300080203881,
     55.901775524173004, 21.41313034223545, 58.1921878299309, 43.73074714833918, 50.60067653177184, 19.045617233572067, 55.71454039291615,
     47.0272218651935, 50.57835753673863, 39.285241149356935, 54.390040159493026, 41.11063278705892, 52.75630252843107, 52.17056237562114,
     60.186941159047834, 35.67631737063206, 59.07573734539187, 29.66288101938385, 64.74900310965913, 43.08416654948153, 51.81327189344188,
     26.447708409086204,18.60266134078831, 18.61484861300581, 19.91288718527447, 18.5513705408779, 17.068467435520724,
    52.97444417777079, 61.68250367122655, 17.793946763667115, 57.32992718367135, 57.83076771609932,
    30.41517193295935, 30.0660872587162, 9.701639270158717, 19.254455342875342, 17.36455404913576,
    17.18676308904514, 17.921802317968424, 16.579405735148352, 17.56109610476716, 55.45048249834662,
    56.07897206020688, 17.43550878328168, 62.887512993653175, 65.36711942546481, 26.97869280982842,
    25.285387678064833, 13.487008294953009, 9.64268405964709, 18.79687790151305, 17.493908753001794,
    18.22016392820857, 19.492441533473905, 15.076039015302987, 60.20724751673406, 53.423315013239346,
    14.90177148106097, 60.971266398818024, 78.54789397429737, 16.27529360716145, 16.46655128997021,
    14.568581784924737, 3.8498461852223143, 17.05987337608919, 16.17329328034001, 18.478592053138414,
    17.607474886160396, 17.508454004745108, 60.152973778315626, 62.12641951129344, 18.01148597163087,
    54.8445431206006, 61.838100621098874,63.08259129116956, 28.554630358715336, 15.918301265581254, 3.3236363092319854,
    16.252557724344324, 15.424844128069996, 18.737197721310277, 15.648255812448147, 7.671786515225004,
    63.07727406599274, 60.38211907504823, 7.629963314542667, 64.28716845089554, 66.49799422765808,
    33.523112466702045, 32.26205481664942, 15.971211207891832, 3.148515568355651, 17.31888289968824,
    16.46181734056367, 19.300773220306116, 19.471925173577137, 14.555893648583751, 64.00289640099479,
    57.88024457035105, 14.768778955504525, 60.19841340646751, 63.26536243245143, 66.11182411385637,
    32.95942085561814, 16.40477124247763, 3.435987781505184, 18.01391987251811, 16.759596640061993,
    21.213637871958312, 18.38503368566234, 13.013117790839782, 66.68599823987624, 62.21836612168693,
    13.21967954477483, 60.1877773778784, 65.12589001050536, 67.87876487650409, 33.765446133986906,
    36.82333651687581, 15.870570252144185, 4.853394437024072, 17.75694837463209, 17.106589315914015,
    22.55418794379689, 18.067299084657982, 10.148190848514023, 63.353948001158834, 63.32697383448194,
    10.274831774903529, 64.21642428274629, 69.30199816634465, 28.679506793184622, 28.426982361343626,
    25.87272122275806, 15.505510535585767, 4.189915914340397, 17.993641698604645, 18.42229651906402,
    22.890635733147, 18.368767960613848, 12.356312936576861, 61.0047227733012, 65.01235272240726,
    12.266985413320969, 63.991242695903566, 62.381021591517325, 28.17193876766879, 28.248262523084225,
    26.130467928584505, 17.11606672142954, 4.7004766720190805, 18.43611084027163, 19.162682729530154,
    21.27569264429917, 18.545768096773156, 14.824852283125844, 63.111152351021154, 55.89428170919655,
    14.659816994331516, 62.89803730310977, 73.41880659719924, 31.960193520407266, 30.683933589074332,
    28.69149556679785, 14.774538927370813, 2.2873761334917715, 16.632424416285538, 17.086526965199575,
    20.379676494613886, 13.442893121334597, 14.042351233311702, 63.85463775147277, 58.94495169166197,
    14.094625975977882, 63.87877388130444, 62.29185278775509, 20.579536009108217, 20.794035138364954,
    18.207211610198705, 7.712544035149025, 2.8589400571560217, 14.331087722751784, 14.378576088442534,
    10.869603633378079, 6.24507201608329, 5.508744641359731, 62.70445033148597, 56.14358612601352,
    5.541160254832595, 61.748906281901746, 62.748906281901746, 17.67176363790785, 17.046283032746885, 15.289197746987407,
    3.6125405397915236, 4.541202749887692]
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