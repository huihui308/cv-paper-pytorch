"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import torch.utils.data as data
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 data_dir=None,
                 img_size=640,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, 
                 color_augment=None,
                 target_transform=VOCAnnotationTransform(),
                 mosaic=False,
                 mixup=False):
        self.root = data_dir
        self.img_size = img_size
        self.image_set = image_sets
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform
        self.mosaic = mosaic
        self.mixup = mixup
        self.color_augment = color_augment
        if self.mosaic:
            print('use Mosaic Augmentation ...')
        if self.mixup:
            print('use MixUp Augmentation ...')


    def __getitem__(self, index):
        im, gt, h, w, scale, offset = self.pull_item(index)
        return im, gt


    def __len__(self):
        return len(self.ids)


    def load_img_targets(self, img_id):
        # load an image
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        # laod a target
        target = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        return img, target, height, width


    """
        当我们获得了一张马赛克图片后，由于这本身就已经是一次比较强的增强，那么我们没必要再去做随机剪裁了，因此，这里我们只需要调用color_augment去做一次色彩空间变换（包括随机水平翻转）即可。而对于非马赛克增强，我们仍采用标准的数据增强操作：随机水平翻转、随机剪裁、色彩空间变换。
        当然，训练过程中，我们仍会采用多尺度训练
    """
    def load_mosaic(self, index):
        # 首先，我们读取一张图像，然后再随机读取另外三张图像
        ids_list_ = self.ids[:index] + self.ids[index+1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        img_lists = []
        tg_lists = []
        # load image and target
        # 将读取进来的图像以及标签信息分别存放在img_lists和tg_lists变量中
        for id_ in ids:
            img_i, target_i, _, _ = self.load_img_targets(id_)
            img_lists.append(img_i)
            tg_lists.append(target_i)

        """
            准备好一张空白的图像。假设通常情况下我们是将每张图像resize到640×640，那么由于我们后续要把四张图片按“田”字格的方式去拼接，因此，这里预先准备好的空白马赛克图片的尺寸为1280×1280，完成整个马赛克拼接过程后，我们会最终将其resize到640×640。另外，我们预先在这张空白的马赛克图片上的每一处像素值均填为像素均值。最后，选择一个马赛克中心点，后续我们会根据这个中心点，将四张图像拼接上去
        """
        mean = np.array([v*255 for v in self.transform.mean])
        mosaic_img = np.ones([self.img_size*2, self.img_size*2, img_i.shape[2]], dtype=np.uint8) * mean
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]
        # yc = xc = self.img_size

        mosaic_tg = []
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            target_i = np.array(target_i)
            h0, w0, _ = img_i.shape

            # 首先单独调整每张图片的大小
            # resize
            scale_range = np.arange(50, 210, 10)
            s = np.random.choice(scale_range) / 100.    # 尺度缩放系数，随机将图片缩放0.5-2.0倍

            if np.random.randint(2):
                # keep aspect ratio
                r = self.img_size / max(h0, w0)
                if r != 1: 
                    img_i = cv2.resize(img_i, (int(w0 * r * s), int(h0 * r * s)))
            else:
                # not keep aspect ratio
                img_i = cv2.resize(img_i, (int(self.img_size * s), int(self.img_size * s)))
            h, w, _ = img_i.shape
            """
                在上面的实现中，我们会随机将每张图片缩放0.5-2.0倍，缩放0.5-1.0倍可以有效增加小目标样本的数量，而1.0-2.0是弥补马赛克带来的对大目标样本数量抑制的负面效应。这个缩放范围并不是固定的，比如在YOLOX中，其设定为0.1-2.0。缩放的过程是和resize一起的，并且，我们会随机确定是否保留长宽比，这一点是笔者自己加进去的，在官方中的实现是没有考虑进来的。笔者任何，不保留长宽比，可以用来模拟一些物体出现不寻常比例的特殊情况，增加样本的丰富性。
            """

            # 处理好四张图片后，我们就围绕着先前已经选好的马赛克中心点将它们拼接在一起
            # # 围绕确定好的马赛克中心点来拼接四张图片
            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            # 并且，根据每张图片所摆放的位置，对标签信息进行调整
            padw = x1a - x1b
            padh = y1a - y1b

            # 调整标签信息
            # labels
            target_i_ = target_i.copy()
            if len(target_i) > 0:
                # a valid target, and modify it.
                target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                target_i_[:, 3] = (h * (target_i[:, 3]) + padh)     
                # check boxes
                valid_tgt = []
                # 我们把太小的边界框删除掉
                for tgt in target_i_:
                    x1, y1, x2, y2, label = tgt
                    bw, bh = x2 - x1, y2 - y1
                    """
                        由于在拼接图象的过程中，由于缩放的缘故，可能会有一些物体的标签信息过于极端，因此，笔者这里加了一段剔除的代码，只保留宽高大于5的目标，小于这个范围的我们都舍弃掉，尽可能不引入目标太小的标签信息。
                    """
                    if bw > 5. and bh > 5.:
                        valid_tgt.append([x1, y1, x2, y2, label])
                # 检查是否有标签
                if len(valid_tgt) == 0:
                    valid_tgt.append([0., 0., 0., 0., 0.])

                mosaic_tg.append(target_i_)
        # 由于拼接过程中，可能某些图片的部分区域超出了马赛克图片的范围，因此，我们还需要再做一次剪裁
        # check target
        if len(mosaic_tg) == 0:
            mosaic_tg = np.zeros([1, 5])
        else:
            mosaic_tg = np.concatenate(mosaic_tg, axis=0)
            # Cutout/Clip targets
            np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
            # normalize
            mosaic_tg[:, :4] /= (self.img_size * 2) 

        return mosaic_img, mosaic_tg, self.img_size, self.img_size


    """
        总结一下在我们所要实现的YOLOv4中的数据预处理流程（假定启动马赛克增强--mosaic）：

        1.读取当前的一张图像 I1；

        2.若使用马赛克增强（np.random.randint(2) == 1），则再随机读取三张图片 I2, I3, I4，对四张图片做随机缩放处理，然后拼接在一起，并处理标签信息；然后，对马赛克图片做随机水平翻转+色彩空间变换的增强操作；

        3.若不使用马赛克增强（np.random.randint(2) == 0），则将当前图片 I1 resize到固定尺寸（如最长边等于640），然后使用随机水平翻转+随机剪裁+色彩空间变换来处理该图片；

        4.最后，将一批图片拼接为一批数据，倘若使用多尺度训练策略，我们采用插值的方法将这一批图片调整至其他尺寸（如320、352、384、……608、640）。

        参考YOLOX的建议，在训练的最后15epoch，我们会关掉马赛克增强，这是考虑到在马赛克增强的过程中，由于存在缩放和剪裁的操作，会留下一些质量很差的样本，最后的15epoch就是为了缓解这一问题。经过笔者的实测，确实能够提升性能，不过小目标的性能会略微有所损失，比如在笔者某次训练中，小目标的AP达到了27.6，但由于最后15epoch关闭了马赛克增强，小目标的AP掉到了26.9。不过，由于总的AP提升了0.6个点，这一策略还是有效的。
    """
    def pull_item(self, index):
        # load a mosaic image
        if self.mosaic and np.random.randint(2):
            # mosaic
            img, target, height, width = self.load_mosaic(index)

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if self.mixup and np.random.randint(2):
                img2, target2, height, width = self.load_mosaic(np.random.randint(0, len(self.ids)))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                target = np.concatenate((target, target2), 0)

            # augment
            img, boxes, labels, scale, offset = self.color_augment(img, target[:, :4], target[:, 4])

        # load an image and target
        else:
            img_id = self.ids[index]
            img, target, height, width = self.load_img_targets(img_id)
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)
            # augment
            img, boxes, labels, scale, offset = self.transform(img, target[:, :4], target[:, 4])
            
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, target, height, width, scale, offset


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    from transforms import TrainTransforms, ColorTransforms, ValTransforms

    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img_size = 640
    dataset = VOCDetection(
                data_dir='d:/datasets/VOCdevkit/',
                img_size=img_size,
                transform=ValTransforms(img_size),
                color_augment=ColorTransforms(img_size),
                mosaic=True,
                mixup=True)
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))
    for i in range(len(dataset)):
        image, target, _, _, _, _ = dataset.pull_item(i)
        image = image.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)]
        image = ((image * std + mean)*255).astype(np.uint8)
        image = image.copy()

        for box in target:
            x1, y1, x2, y2, cls_id = box
            cls_id = int(cls_id)
            color = class_colors[cls_id]
            # class name
            label = VOC_CLASSES[cls_id]
            x1 *= img_size
            y1 *= img_size
            x2 *= img_size
            y2 *= img_size
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
