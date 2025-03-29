import os
import torch

import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path


import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from clip import clip
import utils
from utils import cosine_lr_schedule, print_params_and_flops
from data import create_dataset, create_sampler, create_loader
from ID_pruner_clip import *
import io
import math

from torch.cuda.amp import autocast as autocast



def train(model, data_loader, optimizer, epoch, device, config, interval=50):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    len_data_loader = len(data_loader)
    total_steps = len_data_loader * config['max_epoch']
    global_step=len_data_loader*epoch+1
    PLATON = Pruner(model, args=args, total_step=total_steps, use_no_mask=False, pruner_name=args.pruner_name)

    for i, (image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # start_time=time.time()
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        ID = [3.8801185730222896, 9.41062834645045, 16.70802361350472, 3.983060585480298, 15.931427951369688,
              23.469010456847734, 4.337724734829484, 27.402630106414342, 58.17209819600113, 9.412367988324876,
              28.844461841580106, 82.87956099767047, 10.917573017028591, 42.86343005477394, 230.15449388452947,
              6.025140388594886, 22.70826996423339, 90.99505204297958, 14.844287563132848, 26.62264274005391,
              187.2821229385766, 8.48176908641599, 19.484239515422765, 80.3288539747654, 7.195902217502301,
              16.80901884454787, 55.29282225057416, 6.648403015934508, 15.155885096421637, 33.10895733149114,
              6.3950981294200036, 14.410649133137477, 25.3173245582958, 6.832006226273054, 14.42967308001568,
              21.15525105166163, 6.244548713515252, 14.567018512783829, 20.42848996428503, 6.245272555228624,
              13.708870717089866, 20.564293070476616, 5.831417089803788, 13.750028915888304, 21.66194973632266,
              5.2499847644451165, 14.100964266757705, 23.755879543740818, 5.102757649679363, 13.747536808930693,
              21.22487997429917, 4.798651390728442, 14.088909596325895, 21.99811406038706, 4.911883988248421,
              14.213893955330954, 22.4113391735559, 4.721891221447442, 14.443418165537617, 24.20606360077574,
              4.73962350104258, 14.733469498078742, 21.314970068922403, 4.486924037043091, 16.229332532431933,
              23.957639065461127, 4.841298057241158, 16.267340626791952, 22.53336119073385, 5.178037815321639,
              16.85037513507607, 22.6332709560431, 11.475710262443997, 18.384634561311096, 17.494206218888795,
              13.535304995864667, 22.126841001008337, 22.019347176495764, 8.636503687055848, 16.970703657500806,
              18.045195817566697, 6.735044081741226, 13.568573654268091, 13.453024647420943, 6.481616854892033,
              11.54418228541807, 11.034481120766976, 6.216085581242679, 10.868408874744194, 10.572880808892506,
              6.104757743169182, 10.164374755316413, 9.997478541073331, 5.998121489103896, 10.226309001648719,
              10.287310717352948, 6.051318609142344, 9.950415259452523, 9.981786452002979, 6.255381664003275,
              9.611837933899125, 9.839832673037066, 6.016082345670775, 9.47377050458093, 9.611985836948868,
              5.795208503333999, 8.258925965752358, 8.080369952682496]

        ID=torch.tensor(ID)
        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))


        loss = model(image, caption, alpha=alpha, idx=idx)

        optimizer.zero_grad()
        loss.backward()
        # print('loss.backward()')
        if (global_step % len(data_loader) != 0):
            optimizer.step()
            # print('optimizer.step')
        if(global_step % len(data_loader) == 0):
            modelPath = args.result_dir + '/models/'
            Path(modelPath).mkdir(parents=True, exist_ok=True)
            PATH = modelPath + 'epoch_{}globalstep_{}.pth'.format(
                epoch, global_step)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr_schedule': optimizer.param_groups[0]["lr"],
            }, PATH)
            print(global_step)
            print("successfully saved")

        PLATON.update_and_pruning(args, model, len_data_loader, global_step, ID, args.useID)
        global_step = global_step + 1
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    print('Computing features for evaluation...')

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        if(i==512):
            break
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenize(text).to(device)
        text_output = model.encode_text(text_input)
        text_embed = text_output / text_output.norm(dim=1, keepdim=True)
        print(text_embed)
        print(text_embed.shape)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for image, img_id in data_loader:
        # if (i == 256):
        #     break
        image = image.to(device)
        image_feat = model.encode_image(image)
        image_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        print(image_embed)
        print(image_embed.shape)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config, client):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    config['pretrained'] = args.pretrained
    config['w_sp_attn'] = args.w_sp_attn / args.world_size
    config['w_sp_mlp'] = args.w_sp_mlp / args.world_size
    config['max_epoch'] = args.epoch
    config['init_lr'] = args.lr
    # config['p'] = args.p
    # if not args.evaluate:
        # print('Target compression ratio: {}%'.format(config['p'] * 100))

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s' % config['dataset'], config, client)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    if args.evaluate and args.pruned:
        print("Creating model for evaluation")
        model, preprocess = clip.load(name=args.pruned, device=device, evaluate=True)
        model.tokenize = clip.tokenize
        model.prune_if_compressed(client,config['pretrained'])
        model = model.to(device)	
    else:

        print("Creating model for training")
        model, preprocess = clip.load(name=config['pretrained'], device=device)
        model.tokenize = clip.tokenize

    model.copy_params()
    print_params_and_flops('retrieval_clip', model, device, config)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device, config)

        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)

        if utils.is_main_process():

            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            print(val_result)

            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                       test_loader.dataset.img2txt)
            print(test_result)

            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             }
                with open(os.path.join(args.result_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.result_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            print("LOG: ", log_stats)

        if args.evaluate:
            break

        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_flickr_clip.yaml')
    parser.add_argument('--output_dir', default='./clip_flickr')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--pruned', default='',type=str, help="Path of the pruned model")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--use_ceph', action='store_true')
    parser.add_argument('--pretrained', default='./clip_large_retrieval_flickr.pth', type=str)
    parser.add_argument('--w_sp_attn', default=(22 / 15) * 8e-3, type=float, help='regularization coefficient for attn')
    parser.add_argument('--w_sp_mlp', default=2e-4, type=float, help='regularization coefficient for mlp')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='number of epoches')
    parser.add_argument('--model_dir', default='test')
    # adaptive pruning
    parser.add_argument('--pruner_name', default='Taylor', type=str,
                        help="[PLATON, Taylor,Magnitude]")
    parser.add_argument('--beta1', default=0.85, type=float, help="beta1 for PLATON.")
    parser.add_argument('--deltaT', default=10, type=int, help="The legnth of local window to reweight EMA.")
    parser.add_argument('--beta2', default=0.95, type=float, help="beta2 for PLATON")
    # pruning schedule
    parser.add_argument('--warmup_steps', default=36250, type=int, help="Warmup steps.")
    parser.add_argument('--initial_threshold', default=1., type=float, help="Initial threshold.")
    parser.add_argument('--final_threshold', default=0.20, type=float, help="Final threshold.")
    parser.add_argument('--initial_warmup', default=0, type=int, help="Initial Warmup for Pruner.")
    parser.add_argument('--final_warmup', default=2, type=int, help="Final Warmup for Pruner.")
    parser.add_argument('--useID', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.result_dir = os.path.join(args.output_dir, args.model_dir)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    client = None
    main(args, config, client)