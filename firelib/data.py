
import os
import random
import numpy as np

import cv2
from torchvision import transforms

from firelib.data_utils import get_dataloader, get_filenames



class FireData():
    def __init__(self, cfg):
        self.cfg = cfg


    def get_trainval_dataloader(self):

        class_names = self.cfg['class_names']
        if len(class_names)==0:
            class_names = os.listdir(self.cfg['train_dir'])
            class_names.sort()
        print("class_names: ", class_names)

        train_data = []
        for i,class_name in enumerate(class_names):
            sub_dir = os.path.join(self.cfg['train_dir'],class_name)
            img_path_list = get_filenames(sub_dir)
            img_path_list.sort()
            train_data += [[p,i] for p in img_path_list]
        random.shuffle(train_data)

        if self.cfg['val_dir'] != '':
            val_data = []
            for i,class_name in enumerate(class_names):
                sub_dir = os.path.join(self.cfg['val_dir'],class_name)
                img_path_list = get_filenames(sub_dir)
                img_path_list.sort()
                val_data += [[p,i] for p in img_path_list]

        else:
            print("val_dir is none, use kflod to split data: k=%d val_fold=%d" % (self.cfg['k_flod'],self.cfg['val_fold']))
            all_data = train_data

            fold_count = int(len(all_data)/self.cfg['k_flod'])
            if self.cfg['val_fold']==self.cfg['k_flod']:
                train_data = all_data
                val_data = all_data[:10]
            else:
                val_data = all_data[fold_count*self.cfg['val_fold']:fold_count*(self.cfg['val_fold']+1)]
                train_data = all_data[:fold_count*self.cfg['val_fold']]+all_data[fold_count*(self.cfg['val_fold']+1):]


        print("Train: %d Val: %d " % (len(train_data),len(val_data)))
        input_data = [train_data, val_data]

        train_loader, val_loader = get_dataloader("trainval", 
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader


    def get_test_dataloader(self):
        data_names = get_filenames(self.cfg['test_path'])
        print("total ",len(data_names))
        input_data = [data_names]
        data_loader = get_dataloader("test", 
                                    input_data,
                                    self.cfg)
        return data_loader


