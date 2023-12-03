# 커스텀 데이터셋 있는 파일

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TinaFaceDataset(Dataset):
    def __init__(self, sep='train'):
        super(TinaFaceDataset, self).__init__()
        self.sep = sep
        assert self.sep == 'train' or self.sep == 'test'
        
        self.data_path = list()
        self.labels = list()
        path = "C:/Users/박성준/OneDrive/Desktop/파기딥/deep-learning/dataset"
        for sep in sorted(os.listdir(path)):
            if self.sep == sep:
                for label in sorted(os.listdir(path+'/'+sep)):
                    for img_path in sorted(os.listdir(path+'/'+sep+'/'+label)):
                        self.data_path.append(path+'/'+sep+'/'+label+'/'+img_path)
                        self.labels.append(label)
                    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])
        
        if self.sep == 'train':
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            return img, self.labels[idx]
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            return img

class ResNetDataset(Dataset):
    def __init__(self):
        super(ResNetDataset, self).__init__()
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass



