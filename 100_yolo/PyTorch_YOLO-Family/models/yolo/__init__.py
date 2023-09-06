from .yolov1 import YOLOv1
from .yolov2 import YOLOv2
from .yolov3 import YOLOv3
from .yolov4 import YOLOv4
from .yolo_tiny import YOLOTiny
from .yolo_nano import YOLONano


# build YOLO detector
def build_model(logger, args, cfg, device, num_classes=80, trainable=False):
    
    if args.model == 'yolov1':
        logger.info('Build YOLOv1 ...')
        model = YOLOv1(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov2':
        logger.info('Build YOLOv2 ...')
        model = YOLOv2(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3':
        logger.info('Build YOLOv3 ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3_spp':
        logger.info('Build YOLOv3 with SPP ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3_de':
        logger.info('Build YOLOv3 with DilatedEncoder ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov4':
        logger.info('Build YOLOv4 ...')
        model = YOLOv4(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolo_tiny':
        logger.info('Build YOLO-Tiny ...')
        model = YOLOTiny(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolo_nano':
        logger.info('Build YOLO-Nano ...')
        model = YOLONano(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    return model
