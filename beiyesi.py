import datetime
starttime = datetime.datetime.now()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2

X = []
Y = []

for i in range(0, 4):
    #遍历文件夹，读取图片
    for f in os.listdir("./RLBD-ML/%s" % i):
        #打开一张图片并灰度化
        Images = cv2.imread("./RLBD-ML/%s/%s" % (i, f))
        image=cv2.resize(Images,(256,256),interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0,1], None, [256,256], [0.0,255.0,0.0,255.0]) 
        X.append(((hist/255).flatten()))
        Y.append(i)
X = np.array(X)
Y = np.array(Y)
#切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
#随机率为100%（保证唯一性可以对比）选取其中的20%作为测试集

from sklearn.preprocessing import binarize 
from sklearn.preprocessing import LabelBinarizer

class ML:
    def predict(self, x):
        #预测标签
        X = binarize(x, threshold=self.threshold)
        #使对数似然函数最大的值也使似然函数最大
        #Y_predict = np.dot(X, np.log(prob).T)+np.dot(np.ones((1,prob.shape[1]))-X, np.log(1-prob).T)
        #等价于  lnf(x)=xlnp+(1-x)ln(1-p)
        Y_predict = np.dot(X, np.log(self.prob).T)-np.dot(X, np.log(1-self.prob).T) + np.log(1-self.prob).sum(axis=1)
        
        return self.classes[np.argmax(Y_predict, axis=1)]
        
class Bayes(ML): 
    def __init__(self,threshold):
        self.threshold = threshold
        self.classes = []
        self.prob = 0.0
        
    def fit(self, X, y):
        
        #标签二值化
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y) 
        self.classes = labelbin.classes_ #统计总的类别，10类
        Y = Y.astype(np.float64)
        
        #转换成二分类问题
        X = binarize(X, threshold=self.threshold)#特征二值化,threshold阈值根据自己的需要适当修改
        feature_count = np.dot(Y.T, X) #矩阵转置，对相同特征进行融合
        class_count = Y.sum(axis=0) #统计每一类别出现的个数
        
        #拉普拉斯平滑处理，解决零概率的问题
        alpha = 1.0
        smoothed_fc = feature_count + alpha
        smoothed_cc = class_count + alpha * 2
        self.prob = smoothed_fc/smoothed_cc.reshape(-1, 1)
        
        return self
        
clf0 = Bayes(2).fit(X_train,y_train) #0.2表示阈值
predictions_labels = clf0.predict(X_test)
print(confusion_matrix(y_test, predictions_labels))
print (classification_report(y_test, predictions_labels))
endtime = datetime.datetime.now()
print (endtime - starttime)
