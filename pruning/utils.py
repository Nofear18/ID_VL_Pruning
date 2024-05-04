import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt


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
    matrix = [0 for i in range(168)]#BLIP has a total of 168 target pruning layers
    for n,p in model.named_parameters():
        if (len(p.data.size()) == 2) and ("embedding" not in n) and ("norm" not in n) and ("pos_embed" not in n) and (
                "patch_embed" not in n) and ("cls_token" not in n) and ("transform.dense" not in n):
            param_this_layer = 1
            for dim in p.data.size():
                param_this_layer *= dim
            total_nb_param += param_this_layer

            # only pruning linear and conv layers
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(p.cpu().data.numpy() == 0)#每层多少等于0的weight
            nb_zero_param += zero_param_this_layer#所有层等于0加起来

            matrix[layer_id - 1] = 100. * zero_param_this_layer / param_this_layer#计算每层比例
            if (layer_id < 49):#vision layer
                v_pruning += zero_param_this_layer * 1.
            all_pruning += zero_param_this_layer  * 1.#总剪枝比例
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
    f=open(os.path.join(args.result_dir+'/pruning_data','{}.txt'.format(args.model_dir)), "a")#结果路径
    f.write("Final pruning rate: {}%".format(pruning_perc) + "\n"+"============================="+"\n")
    f.close()
    return pruning_perc







