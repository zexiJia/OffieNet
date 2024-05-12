# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
import numpy as np
import cv2
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F  


def train_basenet(
    batch_size: int,
    fnet: torch.nn.Module,
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50

    scaler = GradScaler()

    mean_loss = 0

    for step, (batch) in enumerate(data_loader):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        imgs, oid = batch
        imgs = imgs.float().cuda()

        with torch.no_grad():
            ans = fnet((imgs - 0.5) / 0.5)
            ori = ans["ori"]
            enh = ans["enhanceImage"]

        with autocast():
            outputs = model(imgs)
            loss1 = (outputs - enh) ** 2
            loss1 = loss1.mean()
            loss = loss1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_value = loss.item()

        if step % print_freq == 1:
            print("step is {}, loss is {}".format(step, loss_value))

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

        mean_loss += loss_value
    # gather the stats from all processes

    print("epoch is {}, loss is {}".format(epoch, mean_loss / step))
    return mean_loss / step


def train_offienet(
    batch_size: int,
    fnet: torch.nn.Module,
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 25
    loss_func = nn.MSELoss()
    scaler = GradScaler()

    mean_loss = 0

    for step, (batch) in enumerate(data_loader):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        imgs, oid = batch
        imgs = imgs.float().cuda()
        with torch.no_grad():
            ans = fnet((imgs - 0.5) / 0.5)
            ori = ans["ori"]
            enh = ans["enhanceImage"]
            ori = F.interpolate(ori, scale_factor=8, mode='bilinear', align_corners=False) 

        with autocast():
            out_img, out_ori = model(imgs,ori)
            loss1 = (out_img - enh) ** 2
            loss1 = loss1.mean()
            loss2 = (out_ori - ori) ** 2
            loss2 = loss2.mean()
            loss = loss1 + loss2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_value = loss.item()

        
        if step % print_freq == 1:
            print("step is {}, loss is {}".format(step, loss_value))

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

        mean_loss += loss_value
    # gather the stats from all processes

    print("epoch is {}, loss is {}".format(epoch, mean_loss / step))
    return mean_loss / step