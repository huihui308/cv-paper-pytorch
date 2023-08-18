#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""

    https://blog.csdn.net/weixin_40161974/article/details/104901928

"""
import os
import random


trainval_percent = 0.8
train_percent = 0.75
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
num_list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(num_list, tv)
train = random.sample(trainval, tr)

print('\nStart create trainval.txt, test.txt train.txt and val.txt')
ftrainval = open('ImageSets/Main/trainval.txt', 'w+')
ftest = open('ImageSets/Main/test.txt', 'w+')
ftrain = open('ImageSets/Main/train.txt', 'w+')
fval = open('ImageSets/Main/val.txt', 'w+')

for i in num_list:
    name = total_xml[i][:-4] + '\n'
    image_name = './JPEGImages/' + str(name).strip('\n') + '.jpg'
    #print(image_name)
    if not os.path.isfile(image_name):
        print('{} not exist, continue'.format(image_name))
        continue
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print('\nCreate trainval.txt, test.txt train.txt and val.txt success\n')