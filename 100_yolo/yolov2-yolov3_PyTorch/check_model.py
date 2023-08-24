#!/usr/bin/env python
# -*-coding:utf-8 -*-
from data import config
from models.yolov2_r50 import YOLOv2R50


if __name__ == '__main__':
    import argparse, torchsummary

    print("\nYOLOv2R50:")
    cfg = config.yolov2_r50_cfg
    anchor_size = cfg['anchor_size_voc']
    model = YOLOv2R50(device='cpu', input_size=cfg['val_size'], anchor_size=anchor_size)
    torchsummary.summary(model, input_size=(3, 416, 416), batch_size=1, device='cpu')