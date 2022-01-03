import os
import time
import random
from tqdm import tqdm

# handing
import pandas as pd
import numpy as np

import cv2

# torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# optim, scheduler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

# pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# pre-trained models
import timm

# augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# cross-validation
from sklearn.model_selection import StratifiedKFold, GroupKFold

# logger
import wandb

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# warnings
import warnings
warnings.filterwarnings('ignore')


# BC 3,4 의 경우 일반적이지 않은 케이스이므로 학습 시 noise로 작용할 확률이 높다..

class config:
    data_dir = '../data/'
    
    device = device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = 256
    epochs = 35
    lr = 1e-3 # [1e-3, 0.00025]
    batch_size = 32
    val_batch_size = 64
    
    num_workers = 0
    
    k = 5
    training_folds = [0,1,2,3,4]
    # k = 9
    # training_folds = [0,1,2,3,4,5,6,7,8]
    seed = 42

    train_dataset = None
    valid_dataset = None
    
    type = 'all' # 'all', 'BC', 'LT'
    
    # 1~42
    version = 'cc5__b0__each__fc_expand__aug5'
    # image size도 키워서 해보자. 384(안좋음), 512
    # BC_4 : 10, 11, 12, 13 delete
    # BC_7 : 8 (7,8,9 8이 작아짐)
    # clf 로 loss 수정 archface?
    # augmentation
    # tta
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def train_get_transforms():
    return A.Compose([
            # A.Resize(random.randint(config.img_size, config.img_size+128), random.randint(config.img_size, config.img_size+128)),
            # A.RandomCrop(config.img_size, config.img_size),
            # A.crops.transforms.CenterCrop(256, 256, p=1.0),
            # A.Resize(config.img_size, config.img_size),
            # A.crops.transforms.CenterCrop(224, 224, p=1.0),
            # A.OneOf([
            #             A.MotionBlur(blur_limit=5),
            #             A.MedianBlur(blur_limit=5),
            #             A.GaussianBlur(blur_limit=5),
            #             A.GaussNoise(var_limit=(5.0, 30.0))], p=0.8),

            # A.RandomBrightness(limit=0.1, p=0.5),
            # A.RandomContrast(limit=[0.9, 1.1], p=0.5),

            
            A.Transpose(p=0.5),
            A.Rotate(limit=90, interpolation=1, border_mode=4, always_apply=False, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            # A.CoarseDropout(p=0.5),
            # A.Cutout(max_h_size=int(config.img_size * 0.5), max_w_size=int(config.img_size * 0.5), num_holes=1, p=0.8),
            ToTensorV2()
    ])


def valid_get_transforms():
    return A.Compose([
            # A.Resize(config.img_size, config.img_size),
            # A.crops.transforms.CenterCrop(224, 224, p=1.0),
            ToTensorV2()
    ])
    
def random_crop(img, img2):
    h_size = img.shape[0]//2
    seed = np.random.rand()
    if seed<0.2:
        img = img[:h_size, :h_size]
        img2 = img2[:h_size, :h_size]
    elif 0.2<=seed<0.4:
        img = img[:h_size, h_size:]
        img2 = img2[:h_size, h_size:]
    elif 0.4<=seed<0.6:
        img = img[h_size:, :h_size]
        img2 = img2[h_size:, :h_size]
    elif 0.6<=seed<0.8:
        img = img[h_size:, h_size:]
        img2 = img2[h_size:, h_size:]
    elif 0.8<=seed:
        img = img[h_size//2:h_size//2*3, h_size//2:h_size//2*3]
        img2 = img2[h_size//2:h_size//2*3, h_size//2:h_size//2*3]
    return img, img2
    
class PlantDataset(Dataset):
    def __init__(self, config, df, mode, transforms=None):
        self.config = config
        self.before_img_path = df['before_file_path']
        self.after_img_path = df['after_file_path']
        
        self.labels = df['time_delta']
        
        self.mode = mode
        self.transforms = transforms
        
        self.images = []
        
        # print(f'########################### {mode} dataset loader')
        # for image_path in tqdm(self.image_paths):
        #     image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        #     self.images.append(image)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        before_img = cv2.imread(self.before_img_path[idx], cv2.IMREAD_GRAYSCALE) # [COLOR_BGR2RGB, IMREAD_GRAYSCALE]
        after_img = cv2.imread(self.after_img_path[idx], cv2.IMREAD_GRAYSCALE)
        before_img, after_img = before_img/255., after_img/255.
        label = self.labels[idx]
        
        ############# random_crop
        # before_img, after_img = random_crop(before_img, after_img)

        if self.transforms!=None:
            before_img = self.transforms(image=before_img)['image']
            after_img = self.transforms(image=after_img)['image']

        data = {
                    'be_img':torch.tensor(before_img, dtype=torch.float32),
                    'af_img':torch.tensor(after_img, dtype=torch.float32),
                    'label':torch.tensor(label, dtype=torch.float32),
                }
        # category
        # data['label'] -= 1
        # data['label'] = torch.tensor(data['label']).long()
        return data

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
class plModel(pl.LightningModule):
    def __init__(self, config):
        super(plModel, self).__init__()

        self.config = config
        chans = config.valid_dataset[0]['be_img'].shape[0]

        self.before_model = timm.create_model(model_name='tf_efficientnet_b0_ns', pretrained=True, in_chans=chans) 
        self.after_model = timm.create_model(model_name='tf_efficientnet_b0_ns', pretrained=True, in_chans=chans) 
        # [efficientnet_b1_pruned, efficientnet_lite0, resnet34, tf_efficientnet_b0_ns, densenet121, tf_efficientnetv2_s_in21k, tf_mobilenetv3_large_075]

        num_classes = 1
        # num_classes = 42

        for model in [self.before_model, self.after_model]:
            if hasattr(model, "fc"):
                nb_ft = model.fc.in_features
                model.fc = nn.Linear(nb_ft, num_classes)
            elif hasattr(model, "_fc"):
                nb_ft = model._fc.in_features
                model._fc = nn.Linear(nb_ft, num_classes)
            elif hasattr(model, "classifier"):
                nb_ft = model.classifier.in_features
                model.classifier = nn.Linear(nb_ft, num_classes)
            elif hasattr(model, "last_linear"):
                nb_ft = model.last_linear.in_features
                model.last_linear = nn.Linear(nb_ft, num_classes)
        
        self.pooled = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(nb_ft*2, num_classes)
        
        # self.global_pool = GeM(p_trainable=True)
        # self.embedding_size = 512
        # self.neck = nn.Sequential(
        #         nn.Dropout(0.3),
        #         nn.Linear(nb_ft*2, self.embedding_size, bias=True),
        #         nn.BatchNorm1d(self.embedding_size),
        #         nn.PReLU()
        #     )
        self.fc = nn.Sequential(
                                    nn.Dropout(0.3),
                                    nn.Linear(nb_ft*5, 512),
                                    nn.Dropout(0.3),
                                    nn.Linear(512, num_classes),
                                )
        
        ############################################## Loss 
        # self.criterion1 = nn.CrossEntropyLoss()
        self.criterion1 = nn.L1Loss()
        # self.criterion2 = nn.MSELoss()
        
        
    def forward(self, x1, x2):
        # out1 = self.before_model(x1)
        # out2 = self.after_model(x2)
        # out = out2-out1
        
        out1 = self.before_model.forward_features(x1)
        out2 = self.after_model.forward_features(x2)
        
        # out1 = self.global_pool(out1)[:,:,0,0]
        # out2 = self.global_pool(out2)[:,:,0,0]
        
        # x = self.neck(torch.cat([out1, out2], 1))
        # out = self.head(x)
        
        out3 = self.pooled(out1-out2)[:,:,0,0]
        out4 = self.pooled(out2-out1)[:,:,0,0]
        out5 = self.pooled(out2+out1)[:,:,0,0]
        # out6 = self.pooled(out2*out1)[:,:,0,0]
        out1 = self.pooled(out1)[:,:,0,0]
        out2 = self.pooled(out2)[:,:,0,0]
        
        x = torch.cat([out1, out2, out3, out4, out5], 1)
        out = self.fc(x)
        
        return out

    def train_dataloader(self):
        loader = DataLoader(
                            self.config.train_dataset,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
                            self.config.valid_dataset,
                            batch_size=self.config.val_batch_size,
                            num_workers=self.config.num_workers,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True,
                        )
        return loader

    def training_step(self, train_batch, batch_idx):
        pred = self.forward(train_batch['be_img'], train_batch['af_img']).squeeze(1)
        # pred = self.forward(train_batch['be_img'], train_batch['af_img'])
        loss = self.criterion1(pred, train_batch['label'])
        # loss = self.criterion1(pred, train_batch['label'])*0.9 + self.criterion2(pred, train_batch['label'])*0.1
        # pred = torch.sigmoid(pred)
        
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {'loss':loss, 'pred':pred.clone().detach().cpu(), 'label':train_batch['label'].clone().detach().cpu()}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        preds = torch.cat([x['pred'] for x in outputs]).numpy().reshape(-1)
        # preds = torch.argmax(torch.cat([x['pred'] for x in outputs]), 1).numpy().reshape(-1)
        labels = torch.cat([x['label'] for x in outputs]).numpy().reshape(-1)
        
        # preds, labels = np.exp(preds), np.exp(labels)
        mse = np.mean((labels-preds)**2)
        
        self.log("total_train_loss", avg_loss, logger=True)
        self.log("total_train_mse", mse, logger=True)

    def validation_step(self, val_batch, batch_idx):
        pred = self.forward(val_batch['be_img'], val_batch['af_img']).squeeze(1)
        # pred = self.forward(val_batch['be_img'], val_batch['af_img'])
        
        loss = self.criterion1(pred, val_batch['label'])
        # loss = self.criterion1(pred, val_batch['label'])*0.9 + self.criterion2(pred, val_batch['label'])*0.1
        # pred = torch.sigmoid(pred)
        
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"val_loss": loss, 'pred':pred.clone().detach().cpu(), 'label':val_batch['label'].clone().detach().cpu()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        preds = torch.cat([x['pred'] for x in outputs]).numpy().reshape(-1)
        # preds = torch.argmax(torch.cat([x['pred'] for x in outputs]), 1).numpy().reshape(-1)
        labels = torch.cat([x['label'] for x in outputs]).numpy().reshape(-1)
        
        # preds, labels = np.exp(preds), np.exp(labels)
        mse = np.mean((labels-preds)**2)
        # my_table = wandb.Table(columns=['labels', 'preds'], data=[labels, preds])
        # data = [labels, preds]
        # columns = ['labels', 'preds']
        
        self.log("total_val_loss", avg_loss, logger=True)
        self.log("total_val_mse", mse, logger=True)
        # self.log_table('table_key', columns=columns, data=data)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=1e-3)
        
        # scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=1)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)
        # scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosine)
        
        return [optimizer], [scheduler_cosine]
    
    
def main():
    
    df_train = pd.read_csv('../data/train.csv')
    df_train['type'] = df_train['before_file_path'].apply(lambda x: 'BC' if 'BC' in x else 'LT')
    # df_train['before_file_path'] = df_train['before_file_path'].apply(lambda x: x.replace('.png', '_resize256.png'))
    # df_train['after_file_path'] = df_train['after_file_path'].apply(lambda x: x.replace('.png', '_resize256.png'))
    df_train['splits'] = df_train['before_file_path'].apply(lambda x: x.split('adjust/')[-1][:5])# + '_' + df_train['time_delta'].astype(str)
    
    train1 = df_train[df_train['type']=='BC'].reset_index(drop=True)
    train2 = df_train[df_train['type']=='LT'].reset_index(drop=True)
    
    # df_train = df_train[~df_train['splits'].isin(['BC_03', 'BC_04', 'LT_08', 'LT_05'])].reset_index(drop=True)
    print(df_train.splits.value_counts())
    
    
    if config.type=='BC':
        df_train = df_train[df_train['type']=='BC'].reset_index(drop=True)
    elif config.type=='LT':
        df_train = df_train[df_train['type']=='LT'].reset_index(drop=True)
        
    # skf = StratifiedKFold(n_splits=config.k, random_state=config.seed, shuffle=True)
    # n_splits = list(skf.split(df_train, df_train['splits']))
    
    # df_train['time_delta'] = np.log(df_train['time_delta'])
    
    gk = GroupKFold(n_splits=config.k)
    # n_splits = list(gk.split(df_train, y=df_train['time_delta'], groups=df_train['splits']))
    n_splits = list(gk.split(train1, y=train1['time_delta'], groups=train1['splits']))
    n_splits2 = list(gk.split(train2, y=train2['time_delta'], groups=train2['splits']))
    train1['n_fold'] = -1
    train2['n_fold'] = -1
    for i in range(config.k):
        train1.loc[n_splits[i][1], 'n_fold'] = i
        train2.loc[n_splits2[i][1], 'n_fold'] = i
    # df_train['n_fold'] = -1
    # for i in range(config.k):
    #     df_train.loc[n_splits[i][1], 'n_fold'] = i
    # print(df_train['n_fold'].value_counts())
    
    for fold in config.training_folds:
        config.start_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())).replace(' ', '_')
        
        
        logger = WandbLogger(name=f"{config.start_time}_{config.version}_{config.k}fold_{fold}", 
                                     project='dacon-plant', 
                                     config={key:config.__dict__[key] for key in config.__dict__.keys() if '__' not in key},
                                    )
    
        tt = pd.concat([train1[train1['n_fold']!=fold], train2[train2['n_fold']!=fold]]).reset_index(drop=True)
        vv = pd.concat([train1[train1['n_fold']==fold], train2[train2['n_fold']==fold]]).reset_index(drop=True)
        # tt = df_train.loc[df_train['n_fold']!=fold].reset_index(drop=True)#.iloc[:1000]
        # vv = df_train.loc[df_train['n_fold']==fold].reset_index(drop=True)
        print(vv['splits'].value_counts())
        
        train_transforms = train_get_transforms()
        valid_transforms = valid_get_transforms()
        
        config.train_dataset = PlantDataset(config, tt, mode='train', transforms=train_transforms)
        config.valid_dataset = PlantDataset(config, vv, mode='valid', transforms=valid_transforms)
        
        print('train_dataset input shape, label : ', config.train_dataset[0]['be_img'].shape, config.train_dataset[0]['af_img'].shape, config.train_dataset[0]['label'])
        print('valid_dataset input shape, label : ', config.valid_dataset[0]['be_img'].shape, config.valid_dataset[0]['af_img'].shape, config.valid_dataset[0]['label'])
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch') # ['epoch', 'step']
        checkpoints = ModelCheckpoint('model/'+config.version, save_top_k=1, monitor='total_val_mse', mode='min', filename=f'{config.k}fold_{fold}__' + '{epoch}_{total_val_loss:.4f}_{total_val_mse:.4f}')
        
        model = plModel(config)
        trainer = pl.Trainer(
                            max_epochs=config.epochs, 
                            gpus=1, 
                            log_every_n_steps=50,
                            # gradient_clip_val=1000, gradient_clip_algorithm='value', # defalut : [norm, value]
                            # amp_backend='native', precision=16, # amp_backend default : native
                            callbacks=[checkpoints, lr_monitor], 
                            logger=logger
                            )
        
        trainer.fit(model)
        del model, trainer
        wandb.finish()
        # break
    
if __name__ == '__main__':
    seed_everything()
    main()