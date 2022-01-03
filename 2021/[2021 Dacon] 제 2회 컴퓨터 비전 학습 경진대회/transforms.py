import math
import random
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop, RandomAffine
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale, RandomRotation


def get_transform(
        target_size=(512, 512),
        transform_list='horizontal_flip', # random_crop | keep_aspect
        augment_ratio=0.5,
        is_train=True,
        ):
    transform = list()
    # transform_list = transform_list.split(', ')
    # augments = list()


    # transform.append(RandomApply(augments, p=augment_ratio))   
    transform.append(ToTensor())
    transform.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transform)
