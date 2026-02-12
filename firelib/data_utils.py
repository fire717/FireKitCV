import os
import random
import numpy as np
import cv2
import albumentations as A
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from firelib.data_aug import data_aug



class TrainValDataLoader(Dataset):
    def __init__(self, data, cfg, data_aug=None,  transform=None):
        self.data = data
        self.cfg = cfg
        self.data_aug = data_aug
        self.transform = transform

    def __getitem__(self, index):

        img = cv2.imread(self.data[index][0])
   
        img = preprocess_img(img, self.cfg)

        if self.data_aug:
            img = self.data_aug(img)

        if self.transform:
            img = self.transform(img)
        
        y = self.data[index][1]

        y_onehot = F.one_hot(torch.tensor([y], dtype=torch.long), num_classes=self.cfg['num_classes']).float()[0]

        return img, y_onehot, self.data[index]
        
    def __len__(self):
        return len(self.data)


class TestDataLoader(Dataset):

    def __init__(self, data, cfg, transform=None):
        self.data = data
        self.cfg = cfg
        self.transform = transform

    def __getitem__(self, index):

        img = cv2.imread(self.data[index])
        img = preprocess_img(img, self.cfg)

        if self.transform is not None:
            img = self.transform(img)


        return img, self.data[index]

    def __len__(self):
        return len(self.data)


def get_filenames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L


def preprocess_img(img, cfg):
    img = cv2.resize(img, cfg['img_size'])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def get_dataloader(mode, input_data, cfg):

    my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # my_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # my_normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)

    if mode=="test":
        test_loader = torch.utils.data.DataLoader(
                            TestDataLoader(
                                input_data[0],
                                cfg,
                                None,
                                transforms.Compose([
                                    transforms.ToTensor(),
                                    my_normalize
                                ])
                            ), 
                            batch_size=cfg['batch_size'], 
                            shuffle=False, 
                            num_workers=cfg['num_workers'], 
                            pin_memory=True
                        )
        return test_loader

    elif mode=="trainval":
 
        train_loader = torch.utils.data.DataLoader(
                                TrainValDataLoader(
                                    input_data[0],
                                    cfg,
                                    data_aug,
                                    transforms.Compose([
                                        transforms.ToTensor(),
                                        my_normalize,
                                    ])
                                ),
                                batch_size=cfg['batch_size'], 
                                shuffle=True, 
                                num_workers=cfg['num_workers'],
                                pin_memory=True
                            )

        val_loader = torch.utils.data.DataLoader(
                                TrainValDataLoader(
                                    input_data[1],
                                    cfg,
                                    None,
                                    transforms.Compose([
                                        transforms.ToTensor(),
                                        my_normalize
                                    ])
                                ),
                                batch_size=cfg['batch_size'], 
                                shuffle=False, 
                                num_workers=cfg['num_workers'], 
                                pin_memory=True
                            )
        return train_loader, val_loader
