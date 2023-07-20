#!/usr/bin/env python
# -*-coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# you need to download the models to ~/.torch/models
# model_urls = {
#     'zfnet': 'https://download.pytorch.org/models/zfnet-owt-4df8aa71.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = 'zfnet-owt-4df8aa71.pth'


class ZFNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ZFNet, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1), # shape is 110 x 110 x 96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # shape is 55 x 55 x 96

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0), # shape is 26 x 26 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # shape is 13 x 13 x 256

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # shape is 6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.init_weight()  # initialize weight

    def init_weight(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class ZFNetNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ZFNetNew, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1), # shape is 110 x 110 x 96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # shape is 55 x 55 x 96

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0), # shape is 26 x 26 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # shape is 13 x 13 x 256

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # shape is 6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            nn.ReLU(),

            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1)
        )
        self.init_weight()  # initialize weight

    def init_weight(self):
        for layer in self.features:
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
        x = self.features(x)
        return self.classifier(x).squeeze()


def zfnet(pretrained=False, **kwargs):
    """
    ZFNet model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = ZFNet(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['zfnet']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model


def zfnet_new(pretrained=False, **kwargs):
    """
    ZFNetNew model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = ZFNetNew(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['zfnet_new']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model


if __name__ == '__main__':
    import argparse, torchsummary

    parser = argparse.ArgumentParser(description='PyTorch zfnet model')
    parser.add_argument('--type', default='standard', type=str, 
                    help='zfnet type, standard or new(use 1x1 con in full connect insted of linear)')
    parser.add_argument('--num_classes', default=1000, type=int, metavar='N',
                    help='numer of total classed to predict')
    args = parser.parse_args()
    if args.type == "standard":
        print("ZFNet from paper")
        model = zfnet(num_classes=args.num_classes)
    elif args.type == "new":
        print("use 1x1 con in full connect insted of linear")
        model = zfnet_new(num_classes=args.num_classes)
    else:
        print('Input {} not support.'.format(args.type))
        os._exit(0)
    torchsummary.summary(model, input_size=(3, 227, 227), batch_size=1, device='cpu')