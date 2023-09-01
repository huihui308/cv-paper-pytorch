#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
    python3 train.py --cuda -ms
    python3 train.py --cuda -ms -d coco
    python3 train.py --cuda --batch_size 8 --version yolov3 -root /home/david/dataset/detect/VOC/ --dataset voc --multi_scale --mosaic --max_epoch 3
"""
from __future__ import division

import logging.config
import os, signal, argparse
from copy import deepcopy

# ----------------- Torch Components -----------------
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import compute_flops
from utils.yolo_logging import Logger
from utils.term_sig_handle import term_sig_handler

# ----------------- Config Components -----------------
from config import build_dataset_config, build_model_config, build_trans_config

# ----------------- Model Components -----------------
from models.detectors import build_model

# ----------------- Train Components -----------------
from engine import build_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')
    # Basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('-size', '--img_size', default=640, type=int, 
                        help='input image size')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")
    parser.add_argument('--vis_tgt', action="store_true", default=False,
                        help="visualize training data.")
    parser.add_argument('--vis_aux_loss', action="store_true", default=False,
                        help="visualize aux loss.")
    parser.add_argument('--log_dir', default='./logs', type=str, 
                        help='Log dir')

    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on all the GPUs.')

    # Epoch
    parser.add_argument('--max_epoch', default=150, type=int, 
                        help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int, 
                        help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=10, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--no_aug_epoch', default=20, type=int, 
                        help='cancel strong augmentation.')

    # Model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')

    # Dataset
    parser.add_argument('--root', default='/Users/liuhaoran/Desktop/python_work/object-detection/dataset/',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    parser.add_argument('--load_cache', action='store_true', default=False,
                        help='load data into memory.')
    
    # Train trick
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='Multi scale')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Model EMA')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--grad_accumulate', default=1, type=int,
                        help='gradient accumulation')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    #logging.config.fileConfig('./utils/logging.conf')
    #logger = logging.getLogger('fileLog01')
    #print(args.log_dir + "/master.log")
    logger = Logger(name=args.log_dir + "/master.log", log_level=logging.DEBUG)

    logger.info("Setting Arguments.. : {}".format(args))
    logger.info("----------------------------------------------------------")
    # Build DDP
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        logger.info("git:\n  {}\n".format(distributed_utils.get_sha()))
    world_size = distributed_utils.get_world_size()
    logger.info('World size: {}'.format(world_size))

    # Build CUDA
    if args.cuda:
        logger.info('use cuda')
        # cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        logger.info('use cpu')
        device = torch.device("cpu")

    # Build Dataset & Model & Trans. Config
    data_cfg = build_dataset_config(args, logger)
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # Build Model
    model, criterion = build_model(args, model_cfg, device, data_cfg['num_classes'], True)

    return
    # Keep training
    if distributed_utils.is_main_process and args.resume is not None:
        print('keep training: ', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    model = model.to(device).train()
    model_without_ddp = model
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Calcute Params & GFLOPs
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        compute_flops(model=model_copy,
                      img_size=args.img_size,
                      device=device)
        del model_copy
    if args.distributed:
        dist.barrier()

    # Build Trainer
    trainer = build_trainer(args, data_cfg, model_cfg, trans_cfg, device, model_without_ddp, criterion, world_size)

    # --------------------------------- Train: Start ---------------------------------
    ## Eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = model_without_ddp
        trainer.eval(model_eval)

    ## Satrt Training
    trainer.train(model)
    # --------------------------------- Train: End ---------------------------------

    # Empty cache after train loop
    del trainer
    if args.cuda:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, term_sig_handler)
    train()
