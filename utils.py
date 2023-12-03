# 자질부리한거 다 여기로 ㄲ
import os
import torch
from torch import nn
from torchvision import models
from torchvision import transforms

from torch.utils.data import Dataset

from PIL import Image

from torchsummary import summary
from tqdm import tqdm

a = 10