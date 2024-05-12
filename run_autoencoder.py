# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from datasets import build_pretraining_dataset
from engine_for_autoencoder import train_basenet,train_offienet
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from Datasets.ImageFolder import Images, NoiseImages
from FingerNet.FingerNet import FingerNet
from Models.basenet import BaseNet
from Models.offienet import OffieNet
from ckputils import checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser("OffieNet", add_help=False)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--save_ckpt_freq", default=2, type=int)
    parser.add_argument("--distributed", default=True)

    # Model parameters
    parser.add_argument(
        "--model",
        default="OffieNet",
        type=str,
        metavar="MODEL",
        help="BaseNet or OffieNet",
    )

    parser.add_argument(
        "--input_size", default=512, type=int, help="images input size for backbone"
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        metavar="LR",
        help="learning rate (default: 1e-2)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument(
        "--output_dir",
        default="/backup2/zexi/data/offienet",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/home/zexi/MAE_Autoencoder/aeresult",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--layer_decay", type=float, default=0.75)

    return parser.parse_args()




def main(args):
    parser = argparse.ArgumentParser()
    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)

    cudnn.benchmark = True

    if args.model == "BaseNet":
        model = BaseNet()
    elif args.model == "OffieNet":
        model = OffieNet()
        checkpoint_path = "basenet.pth"
        checkpoints = torch.load(checkpoint_path)
        model.basenet.load_state_dict(checkpoints["net_weights"])
    

    # get dataset
    dataset_train = Images(
        [
            "/Datasets/CISL24218/matched/T",
            "/Datasets/CISL25632/matched/T",
            '/Datasets/FVC/FVC2000/raw',
            '/Datasets/FVC/FVC2002/raw',
            '/Datasets/FVC/FVC2006/raw/DB2_a',
            '/Datasets/FVC/FVC2006/raw/DB4_a',

        ],
        None,
        None,
    )

    dataset_train_noise = NoiseImages(
        [
            "/Datasets/CISL24218/matched/T",
            "/Datasets/CISL25632/matched/T",
            '/Datasets/FVC/FVC2000/raw',
            '/Datasets/FVC/FVC2002/raw',
            '/Datasets/FVC/FVC2006/raw/DB2_a',
            '/Datasets/FVC/FVC2006/raw/DB4_a',

        ],
        None,
        None,
    )


    fnet = FingerNet().cuda()
    fnet.eval()
    checkpoint("fingernet.pth", "net_weights").keep(
        "net", True
    ).load_ckp(fnet, strict=False)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = (
            len(dataset_train) // args.batch_size // num_tasks
        )

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_train_noise = torch.utils.data.DataLoader(
        dataset_train_noise,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {} M".format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (total_batch_size * num_training_steps_per_epoch)
    )

    optimizer = torch.optim.SGD(model.parameters(), args.lr) 

    loss_scaler = None

    # utils.load_checkpoint(model=model, optimizer=optimizer, loss_scaler=None)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        if args.model == "BaseNet":
            train_stats = train_basenet(
                args.batch_size,
                fnet,
                model,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=None,
            )
        elif args.model == "OffieNet":
            train_stats = train_offienet(
                args.batch_size,
                fnet,
                model,
                data_loader_train_noise,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=None,
            )
        else:
            print('wrong model !')
        
        if args.output_dir:
            if (epoch) % args.save_ckpt_freq == 0:
                utils.save_checkpoint(model, optimizer, epoch, args.output_dir, it=None)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
