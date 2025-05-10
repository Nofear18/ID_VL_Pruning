import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

blip_coco_EXnameList=["embedding", "norm", "pos_embed", "patch_embed", "cls_token","transform.dense"]
blip_coco_INnameList=[]
blip_nlvr_EXnameList=["embedding", "norm", "pos_embed", "patch_embed", "cls_token", "cls_head"]
blip_nlvr_INnameList=[]
clip_flickr_EXnameList=[ '_m.' ]
clip_flickr_INnameList=['out_proj','c_fc','c_proj']


def is_valid_name(name, include_keywords, exclude_keywords) :

    return (not include_keywords or any(keyword in name for keyword in include_keywords)) and all(
        keyword not in name for keyword in exclude_keywords)

def compute_prune_ratio(args,model,type):
    """
        Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0
    # "embedding", "norm", 'patch_embed'
    layer_id = 0
    v_pruning = 0
    all_pruning = 0
    if type=='blip_coco':
        Layer_num=168
        v_layer_num=48
        EXname=blip_coco_EXnameList
        INname=blip_coco_INnameList
    elif type=='blip_nlvr':
        Layer_num = 222
        v_layer_num = 48
        EXname = blip_coco_EXnameList
        INname = blip_coco_INnameList
    elif type=='clip_flickr':
        Layer_num=108
        v_layer_num = 72
        EXname = blip_coco_EXnameList
        INname = blip_coco_INnameList
    else:
        raise ValueError(f"Invalid type: {type}. Supported types: 'blip_coco', 'blip_nlvr', 'clip_flickr'.")

    matrix = [0 for i in range(Layer_num)]
    for n,p in model.named_parameters():
        if (len(p.data.size()) == 2) and is_valid_name(n,INname,EXname):
            param_this_layer = 1
            for dim in p.data.size():
                param_this_layer *= dim
            total_nb_param += param_this_layer

            # only pruning linear and conv layers
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(p.cpu().data.numpy() == 0)#每层多少等于0的weight
            nb_zero_param += zero_param_this_layer#所有层等于0加起来

            matrix[layer_id - 1] = 100. * zero_param_this_layer / param_this_layer#计算每层比例
            if (layer_id <= v_layer_num):#vision layer
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







