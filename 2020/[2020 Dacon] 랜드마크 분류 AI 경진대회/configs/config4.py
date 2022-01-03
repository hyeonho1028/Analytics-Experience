import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
    "DEBUG" : False,
    "num_workers" : 8,
    
    'gpus':'0',
    'distributed_backend': None,
    'sync_batchnorm': True,

    'gradient_accumulation_steps':4,
    'precision':16,

    'warmup_epo':1,
    'cosine_epo':19,
    'lr': 0.002,
    'weight_decay':1e-4,
    
    'p_trainable': True,
    'crit': "bce",

    'backbone':'tf_efficientnet_b2_ns',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45.0,
    'arcface_m': 0.4,
    'neck': 'option-D',
    'head': 'arc_margin',
    
    'pretrained_weights': None,

    'optim':'sgd',
    "batch_size" : 15,
    "n_splits" : 5,
    "fold" : 0,
    "seed" : 756,
    "device" : "cuda:0",

    "out_dim" : 1049,
    "n_classes" : 1049,

    'class_weights': 'log',
    'class_weights_norm' :'batch',

    "normalization" : "imagenet",

}

args['tr_aug'] = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100),    
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(448, 448),
        A.Cutout(max_h_size=int(448 * 0.3), max_w_size=int(448 * 0.3), num_holes=1, p=0.5),
    ])

args['val_aug'] = A.Compose([
        A.ImageCompression(quality_lower=99, quality_upper=100),    
        A.Resize(448, 448),
    ])