import datetime
starttime = datetime.datetime.now()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2

X = []
Y = []

for i in range(0,4):
    #遍历文件夹，读取图片
    for f in os.listdir("./RLBD-ML/%s" % i):
        #打开一张图片并灰度化
        Images = cv2.imread("./RLBD-ML/%s/%s" % (i, f))
        #cv2.imread()读取图片后已多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定
        image=cv2.resize(Images,(256,256),interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0,1], None, [256,256], [0.0,255.0,0.0,255.0]) 
        X.append(((hist/255).flatten()))
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
#切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
#随机率为100%（保证唯一性）选取其中的20%作为测试集

class KNN:
    def __init__(self,train_data,train_label,test_data):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        
    def knnclassify(self):
        num_train = (self.train_data).shape[0]
        num_test = (self.test_data).shape[0] 
        labels = []
        for i in range(num_test):
            y = []
            for j  in range(num_train):
                dis = np.sum(np.square((self.train_data)[j]-(self.test_data)[i]))
                y.append(dis)
            labels.append(self.train_label[y.index(min(y))])
        labels = np.array(labels)
        return labels
knn = KNN(X_train,y_train,X_test)
predictions_labels = knn.knnclassify()
print(confusion_matrix(y_test, predictions_labels))
print (classification_report(y_test, predictions_labels))
endtime = datetime.datetime.now()
print (endtime - starttime)
