import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir,transform=None,target_transform=None):
        self.data_dir=data_dir
        self.transform=transform
        self.target_transform=target_transform

        lst_data=os.listdir(self.data_dir)

        lst_label=[f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label=lst_label
        self.lst_input=lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self,index):
        label=np.load(os.path.join(self.data_dir,self.lst_label[index]))
        input=np.load(os.path.join(self.data_dir,self.lst_input[index]))
        ##네트워크에 들어가는 데이터는 3차원이어야 한다.

        
        #input=np.true_divide(input,255.0)
       

        if label.ndim==2:
            label=label[:,:,np.newaixs]
        if input.ndim==2:
            input = input[:, :, np.newaixs]

        if self.transform:
            torch.manual_seed(10)
            input=self.transform(np.uint8(input))##to_PILImage는 float type 지원안한다
        if self.target_transform:
            torch.manual_seed(10)
            label = self.target_transform(np.uint8(label))
            ##트랜스폼 함수가 정의되어있다면 통과해야한다.

        data = {'input': input, 'label': label}

        return data

