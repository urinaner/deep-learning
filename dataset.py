# 커스텀 데이터셋 있는 파일

import torch
from torch.utils.data import Dataset

class ResNetDataset(Dataset):
    def __init__(self):
        super(ResNetDataset, self).__init__()
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

class TinaFaceDataset(Dataset):
    def __init__(self):
        super(TinaFaceDataset, self).__init__()
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
