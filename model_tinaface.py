# TinaFace 모델
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace# 이미지 저장 경로

# img = cv2.imread(img_path)# 이미지 확인
detection_models = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']


folder_path = "/path/to/your/folder"

file_list = os.listdir(folder_path)

for file in file_list:
    if file.lower().endswith(".png"):
        # 파일의 경로 생성
        img_path = os.path.join(folder_path, file)
        
        try:
            face = DeepFace.detectFace(img_path=img_path, detector_backend='retinaface')
        
            print(f"얼굴이 감지된 이미지: {img_path}")
        except Exception as e:
            print(f"얼굴 감지 중 오류 발생: {img_path}")
            print(f"에러 메시지: {str(e)}")