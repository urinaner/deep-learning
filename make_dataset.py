import torch
import numpy as np
import os
from PIL import Image 

train_path = "C:/Users/wnslg/Desktop/yj/deep-learning/dataset/train"

test_path = "C:/Users/wnslg/Desktop/yj/deep-learning/dataset/test"

for labels in sorted(os.listdir(train_path)):
    for imgs in sorted(os.listdir(train_path+'/'+labels)):
        print(imgs)