import heapq
import os
import sys
import argparse
import logging
import random
import time
from pathlib import Path

import re
import numpy
import torch
import math
import numpy as np
from pruning.utils import compute_prune_ratio
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import random


def quick_select(nums, k):
    if not nums or k < 1 or k > len(nums):
        return None

    pivot = random.choice(nums)
    low = [x for x in nums if x < pivot]
    high = [x for x in nums if x > pivot]
    equal = [x for x in nums if x == pivot]

    if k <= len(high):
        return quick_select(high, k)
    elif k <= len(high) + len(equal):
        return equal[0]
    else:
        return quick_select(low, k - len(high) - len(equal))

class Pruner(object):
    def __init__(self, model, args, total_step,
                 mask_param_name=['out_proj','c_fc','c_proj'],
                 non_mask_name=[ '_m.' ],
                 use_no_mask=False, pruner_name='PLATON'):  

        self.model = model
        self.config = vars(args)
        self.args = args
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.is_dict = {}
        self.total_step = total_step
        self.mask_param_name = mask_param_name
        self.non_mask_name = non_mask_name
        self.use_no_mask = use_no_mask
        # self.total_step = total_step

        self.pruner_name = pruner_name
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]
        self.deltaT = self.config["deltaT"]

    def whether_mask_para(self, n):
        if not self.use_no_mask:
            if('_m' not in n):
                return any(nd in n for nd in self.mask_param_name )

    def schedule_threshold_comb(self, step: int, length: int):
        # Schedule the ramining ratio
        args = self.args
        total_step = self.total_step
        initial_threshold = self.config['initial_threshold']
        final_threshold = self.config['final_threshold']
        initial_warmup = self.config['initial_warmup']
        final_warmup = self.config['final_warmup']
        warmup_steps = self.config['warmup_steps']
        mask_ind = False
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        else:
            spars_warmup_steps = initial_warmup * warmup_steps  # 5400
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps  # 50*3
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if (step % self.deltaT == 0) or (step % length == 0) else False
        return threshold, mask_ind

    def update_ipt_with_local_window(self, model, global_step):
        # Calculate the sensitivity and uncertainty
        for n, p in model.named_parameters():
            if self.whether_mask_para(n) :
                # print(n)
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.beta2 > 0 and self.beta2 != 1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                if self.pruner_name == 'Magnitude':
                    # Calculate the score of magnitude pruning
                    self.ipt[n] = p.abs().detach()
                elif self.pruner_name == 'Taylor':
                    self.ipt[n] = (p*p.grad).abs().detach()
                elif self.pruner_name == 'PLATON':
                    local_step = global_step % self.deltaT
                    update_step = global_step // self.deltaT
                    if local_step == 0:
                        self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                        if self.beta2 > 0 and self.beta2 < 1:
                            self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                                  (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        elif self.beta2 == 2.:
                            self.exp_avg_unc[n] = (update_step * self.exp_avg_unc[n] + \
                                                   (self.ipt[n] - self.exp_avg_ipt[n]) ** 2) / (update_step + 1)
                        self.ipt[n] = (p * p.grad).abs().detach()
                    else:
                        self.ipt[n] = (self.ipt[n] * local_step + (p * p.grad).abs().detach()) / (local_step + 1)
                else:
                    raise ValueError("Incorrect Pruner Name.")

    def mask_with_threshold(self, model, threshold, ID, useID, global_step):
        # Calculate the final importance score
        is_dict = {}
        i = 0
        j = 0
        for n, p in model.named_parameters():
            # print(n)
            if self.whether_mask_para(n) :
                if self.pruner_name == 'Magnitude' or self.pruner_name == 'Taylor' :
                    if (useID):
                        is_dict[n] = self.ipt[n]*ID[i]
                        if ('bias' in n):
                            i = i + 1
                    else:
                        is_dict[n] = self.ipt[n]

                elif self.pruner_name == 'PLATON':
                    if (useID):

                        if self.beta2 > 0 and self.beta2 < 1:

                            is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n] * ID[i]

                        elif self.beta2 == 1.:
                            is_dict[n] = self.exp_avg_ipt[n] * ID[i]
                        elif self.beta2 == 2.:
                            is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt() * ID[i]
                        else:
                            # Handling the uncepted beta2 to default setting
                            is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs() * ID[i]
                        if ('bias' in n):
                            i = i + 1
                    else:
                        if self.beta2 > 0 and self.beta2 < 1:
                            is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                        elif self.beta2 == 1.:
                            is_dict[n] = self.exp_avg_ipt[n]
                        elif self.beta2 == 2.:
                            is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                        else:
                            # Handling the uncepted beta2 to default setting
                            is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        if ('bias' in n):
                            i = i + 1
                else:
                    raise ValueError("Incorrect Pruner Name.")
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - threshold)))[0].item()

        for n, p in model.named_parameters():
            if self.whether_mask_para(n):

                k_v, k_i = torch.topk(is_dict[n].view(-1), 3, largest=False)  
                k = max((k_v))


                this_threshold = max(k, mask_threshold)
                p.data.masked_fill_(is_dict[n] < this_threshold, 0.0)

        
        return mask_threshold

    def update_and_pruning(self, args,model, length, global_step, ID, useID):
        # Update importance score after optimizer stepping

        self.update_ipt_with_local_window(model, global_step)

        threshold, mask_ind = self.schedule_threshold_comb(global_step, length)

        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(model, threshold, ID, useID, global_step)
        else:
            mask_threshold = None
       
        if (global_step % 3000 == 0) or (global_step % length == 0) or (global_step == 42510):
            P_path = args.result_dir + '/pruning_data'
            Path(P_path).mkdir(parents=True, exist_ok=True)
            f = open(os.path.join(P_path,'{}.txt'.format(args.model_dir)), "a")
            f.write("global_step: {}".format(global_step) + "\n")
            f.write("mask_threshold: {}".format(mask_threshold) + "\n" + "--------------" + "\n")
            f.close()
            compute_prune_ratio(args,model,'clip_flickr')#Supported types: 'blip_coco', 'blip_nlvr', 'clip_flickr'
        return threshold, mask_threshold



