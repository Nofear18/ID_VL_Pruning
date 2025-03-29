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
import random

import numpy as np
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
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
from torch.utils.data import Subset

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


def estimate(X, fraction=0.9, verbose=True):
    # Sort the distance matrix
    sorted_distances = np.sort(X, axis=1, kind="quicksort")

    if verbose:
        print("Sorted distance matrix:")
        print(sorted_distances)

    # Extract first and second nearest neighbors
    first_neighbor = sorted_distances[:, 1]
    second_neighbor = sorted_distances[:, 2]

    if verbose:
        print("First neighbors:", first_neighbor)
        print("Second neighbors:", second_neighbor)

    # Identify degenerate cases
    zero_distances = np.where(first_neighbor == 0)[0]
    equal_neighbors = np.where(first_neighbor == second_neighbor)[0]

    if verbose:
        print(f"Found {zero_distances.shape[0]} elements where r1 = 0.")
        print(f"Found {equal_neighbors.shape[0]} elements where r1 = r2.")
        
    # Exclude degenerate cases
    valid_indices = np.setdiff1d(np.arange(sorted_distances.shape[0]), zero_distances)
    valid_indices = np.setdiff1d(valid_indices, equal_neighbors)

    if verbose:
        print(f"Fraction of valid points: {valid_indices.shape[0] / sorted_distances.shape[0]:.2f}")

    first_neighbor = first_neighbor[valid_indices]
    second_neighbor = second_neighbor[valid_indices]

    # Number of points for linear regression
    npoints = int(np.floor(valid_indices.shape[0]))

    # Define mu and empirical cumulative distribution (Femp)
    mu = np.sort(np.divide(second_neighbor, first_neighbor), kind="quicksort")
    N = valid_indices.shape[0]
    Femp = np.arange(1, N + 1) / N

    # Take logarithms for regression (omit last two elements due to zero issues)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # Perform linear regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    if npoints <= 3:
        raise ValueError("Not enough points for reliable regression.")

    regr.fit(x[:npoints, np.newaxis], y[:npoints, np.newaxis])
    slope = regr.coef_[0][0]

    # Calculate correlation coefficient and p-value
    correlation, pval = pearsonr(x[:npoints], y[:npoints])
    
    return x, y, slope, correlation, pval

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

def set_seed():
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.cuda.manual_seed_all(999)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config, args):
    # change your paths here
    start_time =time.time()
    results_folder = args.Path
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    result_id_filename = join(results_folder, "result_id.txt")
    result_mean_id_filename = join(results_folder, "result_mean_id.txt")
    result_bar_filename = join(results_folder, "result_bar.txt")

    if os.path.exists(result_bar_filename):
        os.unlink(result_bar_filename)
    if os.path.exists(result_id_filename):
        os.unlink(result_id_filename)
    if os.path.exists(result_mean_id_filename):
        os.unlink(result_mean_id_filename)

    # set random seeds
    set_seed()

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
    random_indices = np.random.choice(len(train_dataset), nsamples, replace=False)
    train_dataset= Subset(train_dataset, random_indices)

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[bs] * 3,
        num_workers=[0, 0, 0],
        is_trains=[False, False, False],
        collate_fns=[None, None, None],
    )
    
    if os.path.exists(os.path.join(results_folder, "ID.txt")):
        os.remove(os.path.join(results_folder, "ID.txt"))
    for l, module in enumerate(modules):
        f = open(os.path.join(results_folder, "ID.txt"), "a")
        f.write("module: {}".format(module) + "\n")
        f.close()

    print("block_analysis")
    for l, (module, n) in enumerate(zip(tqdm(modules), names)):
        datas = []
        # for data in all_batches:
        for k, data in tqdm(enumerate(train_loader),total=nsamples//bs):
            if k * bs >= nsamples:
                break
            inputs1, inputs2, img_id = data
            print(img_id)
            hout = []
            def hook(module, input, output):
                # extract representations as matrices of shape (n.nsamples, seq_len, hidden_size).
                datas.append(output.cpu())
            handle = module.register_forward_hook(hook)
            with torch.no_grad():
                out = model(inputs1.to(device), inputs2)
            del out
                 
            handle.remove()

        max_dim1 = max([d.shape[1] for d in datas])
        datas = [expand_dim1(d, max_dim1) for d in datas]
        datas = torch.cat(datas, 0)
        s = datas.shape

        datas = datas.reshape((s[0], -1)).detach().cpu()
        dim, std, n = estimate_ID_2nn(datas, device)

        with open(join(results_folder, "dim.txt"), "a") as f:
            print(f"========={l}===========", file=f)
            print(dim, file=f)
        with open(result_bar_filename, "a") as f:
            print(std, file=f)
        with open(result_id_filename, "a") as f:
            print(dim, file=f)

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
    parser.add_argument('--Path', default='test', help="Paths for storing intermediate and final results")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = yaml.load(
            open("configs/caption_coco.yaml", "r"),
            Loader=yaml.Loader,
            )
    main(config, args)
