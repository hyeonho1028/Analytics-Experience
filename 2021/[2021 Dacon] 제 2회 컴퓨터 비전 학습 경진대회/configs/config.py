import os
import cv2
import albumentations as A
abs_path = os.path.dirname(__file__)

args = {
    "SEED":42,
    "n_folds":5,
    "epochs":25,
    "num_classes":26,
    "input_size":256,
    "batch_size":46,
    "num_workers":0,
    "model":"tf_efficientnet_b5_ns",#tf_efficientnet_b3_ns, dm_nfnet_f0
    "pretrained":True,
    "optimizer":"Adam",
    "scheduler":"Plateau",
    "lr":1e-3,
    "weight_decay":0.0,
    "augment_ratio":0.5,
    "lookahead":False,
    "k_param":5,
    "alpha_param":0.5,
    "patience":3,
    "DEBUG":False,
}

args['trn_transforms'] = A.Compose([ 
                            # A.Transpose(p=0.5),
                            A.HorizontalFlip(p=0.5), 
                            A.VerticalFlip(p=0.5),
                            # A.RandomBrightness(limit=0.2, p=0.75),
                            # A.RandomContrast(limit=0.2, p=0.75), 
                            A.OneOf([
                                A.MotionBlur(blur_limit=5),
                                A.MedianBlur(blur_limit=5),
                                A.GaussianBlur(blur_limit=5),
                                A.GaussNoise(var_limit=(5.0, 30.0)),
                            ], p=0.7),
                            # A.OneOf([
                            #     A.OpticalDistortion(distort_limit=1.0),
                            #     A.GridDistortion(num_steps=5, distort_limit=1.),
                            #     A.ElasticTransform(alpha=3),
                            # ], p=0.7),

                            A.CLAHE(clip_limit=4.0, p=0.7),
                            # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                            # A.Cutout(max_h_size=int(256 * 0.375), max_w_size=int(256 * 0.375), num_holes=1, p=0.7),  
                        ])
args['val_transforms'] = A.Compose([
                            ])