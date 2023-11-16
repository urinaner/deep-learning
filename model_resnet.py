from torchvision import models
import torch
import torch.nn as nn


classes = {0 : 'Heart', 1 : 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

resnet50_pretrained = models.resnet50(pretrained=True)
num_classes = 5
num_ftrs = resnet50_pretrained.fc.in_features
resnet50_pretrained.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device('cuda:0')
#resnet50_pretrained.to(device)

#제대로 가져왔는지 확인
#from torchsummary import summary
#summary(resnet50_pretrained, input_size=(3, 224, 224), device=device.type)

