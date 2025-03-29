import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler

import matplotlib.pyplot as plt
import os


def compute_pruneRatio(args,model):
    """
        Print out prune rate for each layer and the whole network
        """
    total_nb_param = 0
    nb_zero_param = 0
    # "embedding", "norm", 'patch_embed'
    layer_id = 0
    v_pruning = 0
    all_pruning = 0
    matrix = [0 for i in range(108)]
    for n,p in model.named_parameters():
        if (len(p.data.size()) == 2) and (('out_proj' in n)or ('c_fc' in n) or('c_proj'in n))and('_m.' not in n):
            param_this_layer = 1
            for dim in p.data.size():
                param_this_layer *= dim
            total_nb_param += param_this_layer

            layer_id += 1
            zero_param_this_layer = np.count_nonzero(p.cpu().data.numpy() == 0)
            nb_zero_param += zero_param_this_layer

            matrix[layer_id - 1] = 100. * zero_param_this_layer / param_this_layer
            if (layer_id < 72):
                v_pruning += zero_param_this_layer * 1.
            all_pruning += zero_param_this_layer  * 1.
    Ratiostr = ','.join(str(j) for j in matrix)
    f=open(os.path.join(args.result_dir+'/pruning_data','{}.txt'.format(args.model_dir)), "a")
    f.write(Ratiostr+"\n")
    f.close()

    if(all_pruning !=0):
        visual_ratio = 1. * v_pruning / all_pruning

        f=open(os.path.join(args.result_dir+'/pruning_data','{}.txt'.format(args.model_dir)), "a")
        f.write("vision-pruning rate: {}".format(visual_ratio) + "\n" )
        f.close()
        text_ratio = 1. - visual_ratio
    pruning_perc = 100. * nb_zero_param / total_nb_param
    f=open(os.path.join(args.result_dir+'/pruning_data','{}.txt'.format(args.model_dir)), "a")
    f.write("Final pruning rate: {}%".format(pruning_perc) + "\n"+"============================="+"\n")
    f.close()
    return pruning_perc







