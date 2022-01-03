import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
    "DEBUG" : True,
    "num_workers" : 8,
    
    'gpus':'0',
    'distributed_backend': None,
    'sync_batchnorm': True,
    'channels_last':False, 

    'gradient_accumulation_steps':4,
    'precision':16,

    'warmup_epo':1,
    'cosine_epo':19,
    'max_epochs':20,
    'lr': 0.01,
    'weight_decay':1e-4,
    
    'p_trainable': True,

    'crit': "bce",

    'backbone':'tf_efficientnet_b3_ns',
    'embedding_size': 512,
    'pool': 'gem',
    'arcface_s': 45,#45
    'arcface_m': 0.4,
    'head': 'arc_margin',
    'neck': 'option-D',
    
    'pretrained_weights': None,

    'optim':'sgd',
    "batch_size" : 24,
    "n_splits" : 5,
    "fold" : 0,
    "seed" : 0,
    "device" : "cuda:0",

    "out_dim" : 1049,
    "n_classes" : 1049,

    'class_weights': "log",
    'class_weights_norm' :'batch',

    "normalization" : "imagenet",
    "crop_size":448,

}

args['tr_aug'] = A.Compose([ A.LongestMaxSize(512,p=1),
                            A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.RandomCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']), 
                            A.HorizontalFlip(always_apply=False, p=0.5), 
                           ],
                            p=1.0
                            )

args['val_aug'] = A.Compose([ A.LongestMaxSize(512,p=1),
                             A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT,p=1),
                            A.CenterCrop(always_apply=False, p=1.0, height=args['crop_size'], width=args['crop_size']), 
                            ], 
                            p=1.0
                            )