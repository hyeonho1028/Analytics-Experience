import gc
import os 
import argparse
import sys
import time
import cv2
import numpy as np
import pandas as pd
import shutil
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold

from warmup_scheduler import GradualWarmupScheduler

import torch
import torch.nn as nn
from torchvision import transforms

from dataloader import *
from models import *
from trainer import *
from transforms import *
from optimizer import *
from utils import seed_everything, find_th

import warnings
warnings.filterwarnings('ignore')


def main():

    # fix seed for train reproduction
    seed_everything(args.SEED)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)


    # TODO dataset loading
    train_df = pd.read_csv('../data/dirty_mnist_2nd_answer.csv')
    idx = [str(i).zfill(5) for i in train_df['index']]
    image_path = np.array([os.path.join('../data/dirty_mnist_2nd/', f'{i}.png') for i in idx])
    labels = train_df.loc[:, 'a':].values
    print(labels, labels.shape)
    
    # TODO sampling dataset for debugging
    if args.DEBUG: 
        total_num = 1000
        image_path = image_path[:total_num]
        labels = labels[:total_num, :]

    model_path = os.path.join('../models/', time.strftime("%Y-%m-%d_%H:%M:%S")).replace(':', '_').replace('-', '_')
    os.makedirs(model_path, exist_ok=True)
    shutil.copy2('configs/config.py', os.path.join(model_path, 'experiment.py'))
    
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.SEED)
    for fold_num, (trn_idx, val_idx) in enumerate(kf.split(image_path, labels)):

        print(f"fold {fold_num} training starts...")
        train_dataset = PathDataset(image_paths=image_path[trn_idx], 
                                    labels=labels[trn_idx, :], 
                                    transforms=args.trn_transforms
                                    )
        valid_dataset = PathDataset(image_paths=image_path[val_idx], 
                                    labels=labels[val_idx, :], 
                                    transforms=args.val_transforms
                                    )

        train_loader = DataLoader(dataset=train_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers, 
                                    shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False, pin_memory=True)

        # define model
        model = build_model(args, device)

        # optimizer definition
        optimizer = build_optimizer(args, model)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 9)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_cosine)
        # scheduler = build_scheduler(args, optimizer, len(train_loader))
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()

        trn_cfg = {'train_loader':train_loader,
                    'valid_loader':valid_loader,
                    'model':model,
                    'criterion':criterion,
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'device':device,
                    'fold_num':fold_num,
                    'model_save_path':model_path,
                    }

        train(args, trn_cfg)

        del model, train_loader, valid_loader, train_dataset, valid_dataset
        gc.collect()


if __name__ == '__main__':
    from conf import *
    print(args)
    main()
    