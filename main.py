# 메인은 말 안해도 알지?
from utils import *
from dataset import *
from model_resnet import *
# from model_tinaface import *
# from train import *
# from eval import *

# ResNet Test
model = ResNet_50(num_classes = len(classes), freeze_resnet=False)
summary(model, input_size=(3,224,224))

# TinaFaceDataset Test
im1 = TinaFaceDataset(sep='train')[0][0]
# im1 = im1.save("example.jpg") 

# dataset folder -> tinaface model dataset
# 전처리 아직 안함
# tinaface output 확인 후 resnet dataset 만들예정

# 그리고 ResNet50모델파일 다시 만들었는데 선언할때 class 혹은 def 선언해서 클래스 아니면 함수로 넘기는게 좋음(안그래하면 코드 ㅈㄴ 난잡해지고 디버깅하기 빡쎔)
# 앵간한 import, 변수 선언 뭐 이런거는 utils.py에 통일
# argparse 쓸줄 알면 그게 제일 편한데 시간은 없고 나만 아는것 같으니(?) 일단 없이 utils 적극 활용 ㄱㄱ
# 궁금한거 있으면 카톡으로 ㄱㄱ

# pip install numpy
# pip install pandas
# pip install torch
# pip install torchvision
# pip install pillow
# pip install tqdm
# pip install torchsummary
# 위에 이것들 해야 코드 돌아갈거임 미리 해두셈






