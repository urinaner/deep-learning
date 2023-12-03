from utils import *
classes = {0 : 'Heart', 1 : 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

class ResNet_50(nn.Module):
    def __init__(self, num_classes, freeze_resnet=False):
        super(ResNet_50, self).__init__()
        self.pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        if freeze_resnet:
            for child in self.pretrained_resnet.children():
                for param in child.parameters():
                    param.requires_grad = False
        
        num_ftrs = self.pretrained_resnet.fc.in_features
        self.pretrained_resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs//2),
            nn.ReLU(),
            nn.Linear(num_ftrs//2, num_classes)
        )

    def forward(self, input):
        return self.pretrained_resnet(input)
