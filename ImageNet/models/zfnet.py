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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


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


if __name__ == '__main__':
    import torchsummary

    model = zfnet(num_classes=1000)
    torchsummary.summary(model, input_size=(3, 227, 227), batch_size=1, device='cpu')