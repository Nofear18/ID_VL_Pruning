from __future__ import print_function, division
import argparse
import copy
import os
import os.path as path
from os import listdir
from os.path import isfile, join
import pickle
import shutil
import sys
import time
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn import linear_model
from math import sqrt, ceil

import ruamel_yaml as yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

from models.blip import blip_decoder
from data import create_dataset, create_sampler, create_loader



# functions to select checkpoint layers and to determine their depths


def expand_dim1(data, d1):
    if data.shape[1] == d1:
        return data
    shape = list(data.shape)
    shape[1] = d1 - shape[1]
    res = torch.cat([data, torch.zeros(shape)], 1)
    return res

class DataRecorder:
    def __init__(self, root, modules):
        self.modules = {v: k for k, v in enumerate(modules)}
        self.root = root
        self.iter = 0

    def hook(self, module, input, output):
        level_n = self.modules[module]
        file_name = self.out_path(level_n, self.iter)
        data = output.cpu()
        torch.save(data, file_name)

    def out_path(self, level_n, iter):
        return os.path.join(self.root, f"act_{level_n}_{iter}.pth")


def estimate(X, fraction=0.9, verbose=True):

    # sort distance matrix
    Y = np.sort(X, axis=1, kind="quicksort")

    if verbose:
        print("Y")
        print(Y)
    k1 = Y[:, 1]
    if verbose:
        print("k1")
        print(k1)
    # print(k1.shape)
    k2 = Y[:, 2]
    if verbose:
        print("k2")
        print(k2)

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print("Found n. {} elements for which r1 = 0".format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print(
            "Found n. {} elements for which r1 = r2".format(degeneracies.shape[0])
        )
        print(degeneracies)

    good = np.setdiff1d(
        np.arange(Y.shape[0]), np.array(zeros)
    )
    if verbose:
        print(good)

    good = np.setdiff1d(good, np.array(degeneracies))
    # if verbose:
    # ==================================================
    # print(good)
    if verbose:
        print("Fraction good points: {}".format(good.shape[0] / Y.shape[0]))

    k1 = k1[good]
    k2 = k2[good]

    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0]))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind="quicksort")  # u
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    if npoints <= 3:
        print("exception, let's debug")
        # q.d()
    regr.fit(
        x[0:npoints, np.newaxis], y[0:npoints, np.newaxis]
    )
    r, pval = pearsonr(x[0:npoints], y[0:npoints])
    return x, y, regr.coef_[0][0], r, pval


def block_analysis(X, blocks=list(range(1, 21)), fraction=0.9):
    n = X.shape[0]
    dim = np.zeros(len(blocks))
    std = np.zeros(len(blocks))
    n_points = []

    for b in blocks:
        # split indexes array
        idx = np.random.permutation(n)
        npoints = int(np.floor((n / b)))
        idx = idx[0 : npoints * b]
        split = np.split(idx, b)
        tdim = np.zeros(b)
        for i in range(b):
            I = np.meshgrid(split[i], split[i], indexing="ij")
            tX = X[tuple(I)]
            _, _, reg, _, _ = estimate(tX, fraction=fraction, verbose=False)
            tdim[i] = reg
        dim[blocks.index(b)] = np.mean(tdim)
        std[blocks.index(b)] = np.std(tdim)
        n_points.append(npoints)

    return dim, std, n_points


def getDepths(model):
    count = 0
    modules = []
    names = []
    depths = []
    for name, p in model.named_modules():
        if (type(p)==torch.nn.modules.linear.Linear)and('predictions.transform.dense' not in name)and('cls.predictions.decoder'not in name):
            count += 1
            modules.append(p)
            depths.append(count)
            names.append(name)
            print(name, "selected")
        else:
            print(name, "no selected")
    depths = np.array(depths)
    return modules, names, depths



def estimate_ID_2nn(data, device="cpu"):
    s = data.shape
    distances = torch.pdist(data.to(device)).cpu().numpy()
    distance_matrix = squareform(distances)
    same_points = np.nonzero((distance_matrix==0).astype("int") * (1-np.eye(s[0])))
    if len(same_points[0]) != 0:
        print(same_points)

    dim, std, n = block_analysis(distance_matrix, list(range(1, 21)), 0.9)

    return np.average(dim), std[19], n


def main(config, args):
    # change your paths here
    start_time =time.time()
    results_folder = args.Path
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    result_id_filename = join(results_folder, "result_id.txt")
    result_mean_id_filename = join(results_folder, "result_mean_id.txt")
    result_bar_filename = join(results_folder, "result_bar.txt")
    dataloader_path = join(results_folder, "dataload")
    if not os.path.exists(dataloader_path):
        os.mkdir(dataloader_path)
    if os.path.exists(result_bar_filename):
        os.unlink(result_bar_filename)
    if os.path.exists(result_id_filename):
        os.unlink(result_id_filename)
    if os.path.exists(result_mean_id_filename):
        os.unlink(result_mean_id_filename)

    # random generator init
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)

    # parameters
    i = 0
    bs = 16
    nsamples = ceil(args.nsamples / bs) * bs
    print(f"using {nsamples}")
    print("Instantiating pre-trained model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])
    if args.cpu:
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # this switch to evaluation mode your network: in this way dropout and batchnorm
    # no more active and you can use the network as a 'passive' feedforward device;
    # forgetting this produces catastrophically wrong results (I kdatetime.now because I did it)
    model.eval()
    print("Training mode : {}".format(model.training))

    modules, names, depths = getDepths(model)

    # images preprocessing methods

    for i, name in enumerate(names):
        print(i, name)

    samplers = [None, None, None]
    train_dataset, val_dataset, test_dataset = create_dataset("caption_coco", config)

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[bs] * 3,
        num_workers=[0, 0, 0],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )
    rcdr = DataRecorder(dataloader_path, modules)
    for l, module in enumerate(modules):
        f = open(os.path.join(results_folder, "ID.txt"), "a")
        f.write("module: {}".format(module) + "\n")
        f.close()
        handle = module.register_forward_hook(rcdr.hook)

    if not args.s:
        print("extract")
        pbar = tqdm(total=nsamples)
        for k, data in enumerate(train_loader, 0):
            if k * bs >= nsamples:
                break
            pbar.update(bs)
            rcdr.iter = k
            inputs1, inputs2, img_id = data
            out = model(inputs1.to(device), inputs2)
            del out
        pbar.close()
    del model
    del train_loader
    torch.cuda.empty_cache()

    print("block_analysis")
    for l, (module, n) in enumerate(zip(tqdm(modules), names)):
        datas = [torch.load(rcdr.out_path(l, k)) for k in range(rcdr.iter + 1)]
        max_dim1 = max([d.shape[1] for d in datas])
        datas = [expand_dim1(d, max_dim1) for d in datas]
        Out = torch.cat(datas, 0)
        s = Out.shape
        Out = Out.reshape((s[0], -1)).detach().cpu()
        dim, std, n = estimate_ID_2nn(Out, device)

        with open(join(results_folder, "dim.txt"), "a") as f:
            print(f"========={l}===========", file=f)
            print(dim, file=f)
        with open(result_bar_filename, "a") as f:
            print(std, file=f)
        with open(result_id_filename, "a") as f:
            print(f"{np.average(dim)},", file=f)
        del Out
    end_time=time.time()
    print('Tatal spend time:{}h'.format((end_time-start_time)/3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='ProgramName',
            description='What the program does',
            epilog='Text at the bottom of help')
    parser.add_argument('-n', '--nsamples', type=int, default=600,help='Number of samples selected')
    parser.add_argument('--gpu', default="0", help="gpu to use. NOTE, even in CPU mode, this code still need around 2GB of GPU memory")
    parser.add_argument('--cpu', action='store_true', help="CPU mode")
    parser.add_argument('-s', action='store_true', help="skip data extract")  # on/off flag
    parser.add_argument('--Path', default='test', help="Paths for storing intermediate and final results")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = yaml.load(
            open("configs/caption_coco.yaml", "r"),
            Loader=yaml.Loader,
            )
    main(config, args)
