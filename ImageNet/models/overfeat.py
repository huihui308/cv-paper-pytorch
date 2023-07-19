#!/usr/bin/env python
# -*-coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# you need to download the models to ~/.torch/models
# model_urls = {
#     'overfeatnet': 'https://download.pytorch.org/models/overfeatnet-owt-4df8aa71.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = 'overfeatnet-owt-4df8aa71.pth'


class OverFeat_fast(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # train with 221x221 5 random crops and their horizontal filps
        # mini- batches of size 128
        # initialized weight randomly with mu=0, sigma=1x10^-2
        # SGD, momentum=0.6, l2 weight decay of 1x10^-5
        # learning rate 5x10^-2, decay by 0.5 after (30, 50, 60, 70, 80) epochs
        # Dropout on FCN?? -> dropout before classifier conv layer

        self.feature_extractor = nn.Sequential(
            # no contrast normalization is used
            # max polling with non-overlapping
            # 1st and 2nd layer stride 2 instead of 4

            # 1st
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 56 x 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 96 x 28 x 28)

            # 2nd
            nn.Conv2d(96, 256, 5, stride= 1),  # (b x 256 x 24 x 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 256 x 12 x 12)

            # 3rd
            nn.Conv2d(256, 512, 3, padding=1),  # (b x 512 x 12 x 12)
            nn.ReLU(),

            # 4th
            nn.Conv2d(512, 1024, 3, padding=1),  # (b x 1024 x 12 x 12)
            nn.ReLU(),

            # 5th
            nn.Conv2d(1024, 1024, 3, padding=1),  # (b x 1024 x 12 x 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 1024 x 6 x 6)
        )

        # fully connecyed layers implemented as a convolution layers
        self.classifier = nn.Sequential(
            # 6th
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(in_channels=1024, out_channels=3072, kernel_size=6),
            nn.ReLU(),

            # 7th
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(3072, 4096, 1),
            nn.ReLU(),

            # 8th
            nn.Conv2d(4096, num_classes, 1)
        )

        self.init_weight()  # initialize weight

    def init_weight(self):
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.feature_extractor(x)
        return self.classifier(x).squeeze()


class OverFeat_accurate(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # train with 221x221 5 random crops and their horizontal filps
        # mini- batches of size 128
        # initialized weight randomly with mu=0, sigma=1x10^-2
        # SGD, momentum=0.6, l2 weight decay of 1x10^-5
        # learning rate 5x10^-2, decay by 0.5 after (30, 50, 60, 70, 80) epochs
        # Dropout on FCN?? -> dropout before classifier conv layer

        self.feature_extractor = nn.Sequential(
            # no contrast normalization is used
            # max polling with non-overlapping
            # 1st and 2nd layer stride 2 instead of 4

            # 1st
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),  # (b x 96 x 108 x 108)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # (b x 96 x 36 x 36)

            # 2nd
            nn.Conv2d(96, 256, 7, stride= 1),  # (b x 256 x 30 x 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 256 x 15 x 15)

            # 3rd
            nn.Conv2d(256, 512, 3, padding=1),  # (b x 512 x 15 x 15)
            nn.ReLU(),

            # 4th
            nn.Conv2d(512, 512, 3, padding=1),  # (b x 512 x 15 x 15)
            nn.ReLU(),

            # 5th
            nn.Conv2d(512, 1024, 3, padding=1),  # (b x 1024 x 15 x 15)
            nn.ReLU(),

            # 6th
            nn.Conv2d(1024, 1024, 3, padding=1),  # (b x 1024 x 15 x 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # (b x 1024 x 5 x 5)
        )

        # fully connecyed layers implemented as a convolution layers
        self.classifier = nn.Sequential(
            # 7th
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=4096, kernel_size=5),
            nn.ReLU(),

            # 8th
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),

            # 9th
            nn.Conv2d(4096, num_classes, 1)
        )

        self.init_weight()  # initialize weight

    def init_weight(self):
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.feature_extractor(x)
        return self.classifier(x).squeeze()


def overfeatnet_fast(pretrained=False, **kwargs):
    """
    OverFeat_fast model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = OverFeat_fast(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['overfeatnet_fast']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model


def overfeatnet_accurate(pretrained=False, **kwargs):
    """
    OverFeat_fast model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = OverFeat_fast(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['overfeatnet_accurate']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model


if __name__ == '__main__':
    import argparse, torchsummary

    parser = argparse.ArgumentParser(description='PyTorch overfeat model')
    parser.add_argument('--type', default='fast', type=str, 
                    help='overfeat type, fast or accurate')
    parser.add_argument('--num_classes', default=1000, type=int, metavar='N',
                    help='numer of total classed to predict')
    args = parser.parse_args()
    input_size = (3, 231, 231)
    if args.type == "fast":
        print("fast overfeat")
        model = OverFeat_fast(num_classes=args.num_classes)
        input_size = (3, 231, 231)
    elif args.type == "accurate":
        print("accurate overfeat")
        model = OverFeat_accurate(num_classes=args.num_classes)
        input_size = (3, 221, 221)
    else:
        print('Input {} not support.'.format(args.type))
        os._exit(0)
    torchsummary.summary(model, input_size=input_size, batch_size=1, device='cpu')