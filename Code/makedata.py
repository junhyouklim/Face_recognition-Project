from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np

# 분류 대상 카테고리
root_dir = "D:/python Project/face_recognition/FERPlus-master/data/Train"
categories = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
nb_classes = len(categories)
image_size = 48

# 폴더마다의 이미지 데이터 읽어 들이기
X = [] # 이미지 데이터
Y = [] # 레이블 데이터

for idx, cat in enumerate(categories):
    image_dir = root_dir+ "/" + cat
    files = glob.glob(image_dir + "/*.png")
    print("---",cat,"처리중")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB") # 색상 모드 변경
        img = img.resize((image_size, image_size)) # 이미지 크기 변경
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

# 학습 전용 데이터와 테스트 전용 데이터 분류하기
x_train, x_test, y_train, y_test = \
    train_test_split(X,Y)
xy = (x_train, x_test, y_train, y_test)
np.save("image_data.npy",xy)
print("ok", len(Y))
