import time
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QRadioButton,QFileDialog
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
import seaborn as sns
from pandas import DataFrame
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import make_scorer, roc_auc_score
from scipy import stats
from sklearn import preprocessing
from scipy.stats import randint
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot 
import numpy as np
import seaborn as sn
import pathlib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
from keras.layers import Convolution2D, MaxPooling2D	
from  keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
import joblib





class window(QMainWindow):

    def __init__(self):
        super(window, self).__init__()
        loadUi("makine_arayuz.ui", self)
        self.verisecbtn.clicked.connect(self.veriseticek)
        self.egitbtn.clicked.connect(self.egit)
        self.pushButton.clicked.connect(self.rangrid)
        self.kfoldk.currentTextChanged.connect(self.kfoldcb)
        self.baggingbtn.clicked.connect(self.bagging)
        self.votingbtn.clicked.connect(self.voting)
        self.boostingbtn.clicked.connect(self.boosting)
        self.cnnbtn.clicked.connect(self.cnn)
        self.svm_ayir.clicked.connect(self.holdorkfold)
        self.nullbtn.clicked.connect(self.nullclear)
        self.groupBox_8.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_7.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_5.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_9.setStyleSheet("QGroupBox { border: 1px solid red;}")     
        self.groupBox_6.setStyleSheet("QGroupBox { border: 1px solid red;}")    
            
           
    def holdorkfold(self):
        algorithm = self.comboBox_3.currentText()
        if(algorithm=="Hold Out"):
            self.holdout()
        else:
            self.kfold()  
        print("Tamamlandı.")      
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox_7.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
        
        
    def holdout(self):
        if self.over.isChecked():
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            content = self.comboBox.currentText() 
            oversample = RandomOverSampler(sampling_strategy='minority')
            X, y = oversample.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
            
        if self.under.isChecked():
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            content = self.comboBox.currentText() 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            X, y = undersample.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
            
        if(self.comboBox_2.currentText() == "PCA"):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            pca = PCA(n_components=2, svd_solver='full')
            self.XPCA = pca.fit_transform(X)
            self.holdOutBool=True
            self.k_foldBool=False
            content = self.comboBox.currentText() 
            X_train, X_test, y_train, y_test = train_test_split(self.XPCA, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.xtraincekHold(self.HX_train)
            self.xtestcekHold(self.HX_test)
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
            print(X_test.shape)
        elif(self.comboBox_2.currentText() == "chi2"):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            self.Xchi2 = SelectKBest(chi2, k=5).fit_transform(X,y)
            self.holdOutBool=True
            self.k_foldBool=False
            content = self.comboBox.currentText() 
            X_train, X_test, y_train, y_test = train_test_split(self.Xchi2, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.xtraincekHold(self.HX_train)
            self.xtestcekHold(self.HX_test)
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
            print(X_test.shape)
        elif(self.comboBox_2.currentText() == "No Selection"):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            self.holdOutBool=True
            self.k_foldBool=False
            content = self.comboBox.currentText() 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.xtraincekHold(self.HX_train)
            self.xtestcekHold(self.HX_test)
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
            print(X_test.shape)
    
    def xtraincekHold(self,xtrain):
        xtrain=pd.DataFrame(xtrain)
        c=len(xtrain.columns)
        r=len(xtrain.values)
        self.xtraintv.setColumnCount(c)
        self.xtraintv.setRowCount(r)
        for i,row in enumerate(xtrain):
                 for j,cell in enumerate(xtrain.values):
                     self.xtraintv.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
    
    def egit(self):
        if(self.holdOutBool):
            if(self.algseccb.currentText()=="KNN"):
                self.knn(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
            elif(self.algseccb.currentText()=="DT"):
                self.dt(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
            elif(self.algseccb.currentText()=="SVM"):
                self.svm(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
            elif(self.algseccb.currentText()=="LR"):
                self.lr(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
        else:
            if(self.algseccb.currentText()=="KNN"):
                self.knn(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
            elif(self.algseccb.currentText()=="DT"):
                self.dt(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
            elif(self.algseccb.currentText()=="SVM"):
                self.svm(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
            elif(self.algseccb.currentText()=="LR"):
                self.lr(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
        self.groupBox_9.setStyleSheet("QGroupBox { border: 1px solid green;}")     
        self.groupBox_6.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid green;}")
        print("Tamamlandı.")  
    

    
    
    def Ytestcek(self, y_test):
        self.ytesttb.setText(str(y_test.to_string()))
        
    
    
    def Ytraincek(self, y_train):
        self.ytraintb.setText(str(y_train.to_string()))
                     
                     
                     
        
    def xtestcekHold(self,X_test):
        X_test = pd.DataFrame(X_test)
        c=len(X_test.columns)
        r=len(X_test.values)
        self.xtesttv.setColumnCount(c)
        self.xtesttv.setRowCount(r)
        for i,row in enumerate(X_test):
                 for j,cell in enumerate(X_test.values):
                     self.xtesttv.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                     
                     
                     
                     
    def xtestcekKfold(self,X_test):
        X_test = pd.DataFrame(X_test)
        c=len(X_test.columns)
        r=len(X_test.values)
        self.xtesttv.setColumnCount(c)
        self.xtesttv.setRowCount(r)
        for i,row in enumerate(X_test):
                 for j,cell in enumerate(X_test.values):
                     self.xtesttv.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
        
        
        
         
                     
    def xtraincekKfold(self,xtrain):
        xtrain=pd.DataFrame(xtrain)
        c=len(xtrain.columns)
        r=len(xtrain.values)
        self.xtraintv.setColumnCount(c)
        self.xtraintv.setRowCount(r)
        for i,row in enumerate(xtrain):
                 for j,cell in enumerate(xtrain.values):
                     self.xtraintv.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
    
    
    def kfold(self):
        self.k_foldBool=True
        self.holdOutBool=False
        tut=0
        if(self.comboBox_2.currentText() == "PCA"):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            pca = PCA(n_components=2, svd_solver='full')
            self.XPCA = pca.fit_transform(X)
            X = self.XPCA
        elif(self.comboBox_2.currentText() == "chi2"):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            self.Xchi2 = SelectKBest(chi2, k=5).fit_transform(X,y)
            X = self.Xchi2
        elif(self.comboBox_2.currentText() == "No Selection"):
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
        scaler = MinMaxScaler(feature_range=(0, 10))
        X = scaler.fit_transform(X)
        self.Kf=[None, None, None, None, None, None, None, None, None, None]
        cv = KFold(n_splits=5, random_state=None, shuffle=False)
        print(str(X))
        for train_index, test_index in cv.split(X):
            bag=[]
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            bag.append(X_train)
            bag.append(X_test)
            bag.append(y_train)
            bag.append(y_test)
            self.Kf[tut]=bag
            self.kfoldi(tut)
            tut+=1
        if self.over.isChecked():
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            content = self.comboBox.currentText() 
            oversample = RandomOverSampler(sampling_strategy='minority')
            X, y = oversample.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
            
        if self.under.isChecked():
            X = self.data.iloc[:,:-1]
            y = self.data.iloc[:,-1]
            content = self.comboBox.currentText() 
            undersample = RandomUnderSampler(sampling_strategy='majority')
            X, y = undersample.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(content), random_state=10)
            self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
            self.Ytestcek(y_test)
            self.Ytraincek(y_train)
        
    def kfoldi(self,tut):
        if(self.k_foldBool):
            self.KfoldCurrentX_train=self.Kf[tut][0]
            self.KfoldCurrentX_test=self.Kf[tut][1]
            self.KfoldCurrenty_train=self.Kf[tut][2]
            self.KfoldCurrenty_test=self.Kf[tut][3]
            self.xtraincekKfold(self.KfoldCurrentX_train)
            self.Ytraincek(self.KfoldCurrenty_train)
            self.xtestcekKfold(self.KfoldCurrentX_test)
            self.Ytestcek(self.KfoldCurrenty_test)
            
    def kfoldcb(self):
        if(self.k_foldBool):
            self.KfoldCurrentX_train=self.Kf[int(self.kfoldk.currentText())-1][0]
            self.KfoldCurrentX_test=self.Kf[int(self.kfoldk.currentText())-1][1]
            self.KfoldCurrenty_train=self.Kf[int(self.kfoldk.currentText())-1][2]
            self.KfoldCurrenty_test=self.Kf[int(self.kfoldk.currentText())-1][3]
            self.xtraincekKfold(self.KfoldCurrentX_train)
            self.Ytraincek(self.KfoldCurrenty_train)
            self.xtestcekKfold(self.KfoldCurrentX_test)
            self.Ytestcek(self.KfoldCurrenty_test)

        
    def veriseticek(self):
       file_name,_= QFileDialog.getOpenFileName(self, 'Open Image File', r".\Desktop")
       self.data = pd.read_csv(file_name)
       c=len(self.data.columns)
       r=len(self.data.values)
       self.tableWidget.setColumnCount(c)
       self.tableWidget.setRowCount(r)
       colmnames=["ID","SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
       self.tableWidget.setHorizontalHeaderLabels(colmnames)
       for i,row in enumerate(self.data):
             for j,cell in enumerate(self.data.values):
                  self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
        
       self.groupBox_8.setStyleSheet("QGroupBox { border: 1px solid yellow;}")           
       
        
       
    def xtraincek(self,xtrain):
        xtrain = DataFrame(xtrain)
        c=len(xtrain.columns)
        r=len(xtrain.values)
        self.xtraintv.setColumnCount(c)
        self.xtraintv.setRowCount(r)
        colmnames=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
        self.xtraintv.setHorizontalHeaderLabels(colmnames)
        for i,row in enumerate(xtrain):
                 for j,cell in enumerate(xtrain.values):
                     self.xtraintv.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                     self.data.info()
                     
        

        
    def xtestcek(self,X_test):
        X_test = DataFrame(X_test)
        c=len(X_test.columns)
        r=len(X_test.values)
        self.xtesttv.setColumnCount(c)
        self.xtesttv.setRowCount(r)
        colmnames=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
        self.xtesttv.setHorizontalHeaderLabels(colmnames)
        for i,row in enumerate(X_test):
                 for j,cell in enumerate(X_test.values):
                     self.xtesttv.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                     self.data.info()
         

    
        
    def nullclear(self):
        self.data = self.data.dropna(axis=0)
        c=len(self.data.columns)
        r=len(self.data.values)
        self.tableWidget.setColumnCount(c)
        self.tableWidget.setRowCount(r)
        colmnames=["ID","SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
        self.tableWidget.setHorizontalHeaderLabels(colmnames)
        for i,row in enumerate(self.data):
            for j,cell in enumerate(self.data.values):
                      self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                      self.data.info()
        self.groupBox_8.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
    
    
    def knn(self, inf, X_train, X_test, y_train, y_test):   
        if(self.comboBox_2.currentText() == "PCA"):
            X=self.XPCA
        elif(self.comboBox_2.currentText() == "chi2"):
            X=self.Xchi2
        elif(self.comboBox_2.currentText() == "No Selection"):
            X = self.data.iloc[:,0:4] 
            y = self.data.iloc[:,-1] 
            k_range = list(range(1,26))
            scores = []
            for k in k_range:
                knn1 = KNeighborsClassifier(n_neighbors=k)
                knn1.fit(X, y)
                y_pred = knn1.predict(X)
                scores.append(metrics.accuracy_score(y, y_pred))
            plt.plot(k_range, scores)
            plt.xlabel('KNN için K Değeri')
            plt.ylabel('Başarı Sonuç')
            plt.title('KNN İçin Başarı Tablosu')
            plt.show()
            print(metrics.accuracy_score(y, y_pred))
            self.xtraincek(X_train)
            self.ytraintb.setText(str(y_train.to_string()))
            self.xtestcek(X_test)
            self.ytesttb.setText(str(y_test.to_string()))
        k_range = list(range(1,26))
        scores = []
        for k in k_range:
            knn2 = KNeighborsClassifier(n_neighbors=k)
            knn2.fit(X_train, y_train)
            y_pred = knn2.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))
        plt.plot(k_range, scores)
        plt.xlabel('KNN için K Değeri')
        plt.ylabel('Başarı Sonuç')
        plt.title('KNN İçin Başarı Tablosu')
        plt.savefig("KNN.png")
        self.pixmap = QPixmap("KNN.png")
        self.label_11.setPixmap(self.pixmap)
        plt.show()
        prediction=knn2.predict(X_test)
        self.textEdit_cm.setText(str(confusion_matrix(y_test, prediction)))
        acc=(metrics.accuracy_score(y_test, y_pred))*100
        self.acc.setText("ACC : {:.2f}%".format(acc))
        self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
        self.pltRoc(y_test,y_pred,"KNN")
        self.confmat(y_test, y_pred, "KNN")
            
     
        
     
    def svm(self, inf, X_train, X_test, y_train, y_test): 
        if(self.comboBox_2.currentText() == "PCA"):
            X=self.XPCA
        elif(self.comboBox_2.currentText() == "chi2"):
            X=self.Xchi2
        elif(self.comboBox_2.currentText() == "No Selection"):
            X = self.data.iloc[:,0:4] 
            y = self.data.iloc[:,-1] 
        self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
        self.xtraincek(X_train)
        self.ytraintb.setText(str(y_train.to_string()))
        self.xtestcek(X_test)
        self.ytesttb.setText(str(y_test.to_string()))
        iris=self.data
        sns.pairplot(iris)
        sns.pairplot(iris, hue= 'Species')
        sns.set_style('darkgrid')
        setosa = iris[iris['Species']=='Iris-setosa']
        sns.kdeplot(setosa['SepalWidthCm'], setosa['SepalLengthCm'], cmap='plasma', shade=True, shade_lowest=False)
        versicolor = iris[iris['Species']=='Iris-versicolor']
        sns.kdeplot(versicolor['SepalWidthCm'], versicolor['SepalLengthCm'], cmap='plasma', shade=True, shade_lowest=False)
        virginica = iris[iris['Species']=='Iris-virginica']
        sns.kdeplot(virginica['SepalWidthCm'], virginica['SepalLengthCm'], cmap='plasma', shade=True, shade_lowest=False)
        model = SVC()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(classification_report(y_test, prediction))
        self.textEdit_cm.setText(str(confusion_matrix(y_test, prediction)))
        self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
        acc = metrics.accuracy_score(prediction,y_test)
        # df_cm = pd.DataFrame(cm, index = [i for i in np.unique(y)],
        #           columns = [i for i in np.unique(y)])
        # plt.figure(figsize = (5,5))
        # sn.heatmap(df_cm, annot=True)
        # plt.savefig("SVM.png")
        # self.pixmap = QPixmap("SVM.png")
        # self.label_11.setPixmap(self.pixmap)
        # plt.show()
        self.confmat(y_test,prediction,"SVM")
        self.acc.setText("ACC : {:.2f}%".format(acc))
        self.pltRoc(y_test,prediction,"svmroc")
       
        
         
    def lr(self, inf, X_train, X_test, Y_train, Y_test):
        content = self.comboBox.currentText()
        if(self.comboBox_2.currentText() == "PCA"):
            X=self.XPCA
        elif(self.comboBox_2.currentText() == "chi2"):
            X=self.Xchi2
        elif(self.comboBox_2.currentText() == "No Selection"):
             X = self.data.iloc[:,0:4] 
             Y = self.data.iloc[:,-1]
        try:
            self.data.drop("Id",axis=1,inplace=True)
            X = self.data.iloc[:,0:4] 
            Y = self.data.iloc[:,-1]
            self.xtraincek(X_train)
            self.ytraintb.setText(str(Y_train.to_string()))
            self.xtestcek(X_test)
            self.ytesttb.setText(str(Y_test.to_string()))
            log = LogisticRegression()
            log.fit(X_train,Y_train)
            prediction=log.predict(X_test)
            self.pltRoc(Y_test,prediction,"LRROC1")
            self.textEdit_cm.setText(str(confusion_matrix(Y_test, prediction))) 
            self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
            acc = metrics.accuracy_score(prediction,Y_test)
            self.acc.setText("ACC : {:.2f}%".format(acc))
            self.confmat(Y_test, prediction, "LR-CNF")
        except:
            X = self.data.iloc[:,0:4] 
            Y = self.data.iloc[:,-1]
            X.head()
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=float(content),random_state=0)
            self.xtraincek(X_train)
            self.ytraintb.setText(str(Y_train.to_string()))
            self.xtestcek(X_test)
            self.ytesttb.setText(str(Y_test.to_string()))
            log = LogisticRegression()
            log.fit(X_train,Y_train)
            prediction=log.predict(X_test)
            self.pltRoc(Y_test,prediction,"LRROC1")
            self.textEdit_cm.setText(str(confusion_matrix(Y_test, prediction))) 
            self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
            acc = metrics.accuracy_score(prediction,Y_test)
            self.acc.setText("ACC : {:.2f}%".format(acc))
            self.confmat(Y_test, prediction, "LR-CNF")
          
        
        
        
        
    def rangrid(self):
        if(self.comboBox_4.currentText()=="LR" and self.radioButton_2.isChecked()):  
            X = self.data.iloc[:,0:4]
            Y = self.data.iloc[:,-1]
            X.head()
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
            log = LogisticRegression()
            log.fit(X_train,Y_train)
            grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
            logreg=LogisticRegression()
            logreg_cv=GridSearchCV(logreg,grid,cv=10)
            logreg_cv.fit(X_train,Y_train)
            logreg2=LogisticRegression(C=1,penalty="l2")
            logreg2.fit(X_train,Y_train)
            prediction=log.predict(X_test)
            self.confmat(Y_test, prediction, "grid-LR-CNF")
            self.pltRoc(Y_test,prediction,"grid-LRROC1")
            self.textEdit.setText(str(logreg2.score(X_test,Y_test))+ "\n" +str(logreg_cv.best_score_) + "\n " + str(logreg_cv.best_params_))
        elif(self.comboBox_4.currentText()=="LR" and self.radioButton.isChecked()):
            
            X = self.data.iloc[:,0:4]
            Y = self.data.iloc[:,-1]
            X.head()
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
            log = LogisticRegression()
            log.fit(X_train,Y_train)
            logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)
            distributions = dict(C=uniform(loc=0, scale=4),penalty=['l2', 'l1'])
            clf = RandomizedSearchCV(logistic, distributions, random_state=0)
            search = clf.fit(X, Y)
            search.best_params_
            self.textEdit.setText(str(search.best_params_))
            prediction=log.predict(X_test)
            self.confmat(Y_test, prediction, "random-LR-CNF")
            self.pltRoc(Y_test,prediction,"random-LRROC1")
        elif(self.comboBox_4.currentText()=="SVM" and self.radioButton_2.isChecked()):
            X = self.data.iloc[:,0:4] 
            y = self.data.iloc[:,-1]  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = SVC()
            model.fit(X_train, y_train)
            param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
            grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
            grid.fit(X_train, y_train)
            grid_predictions = grid.predict(X_test)
            self.textEdit.setText(str(classification_report(y_test, grid_predictions)))
            self.confmat(Y_test, grid_predictions, "svmgrid-CNF")
            self.pltRoc(Y_test,grid_predictions,"swmgrid-ROC1")
            
        elif(self.comboBox_4.currentText()=="SVM" and self.radioButton.isChecked()):
            self.textEdit.setText("Yapamadım.")
            
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid green;}")
            
            
            
            
        
    def dt(self, inf, X_train, X_test, Y_train, Y_test): 
        content = self.comboBox.currentText()
        if(self.comboBox_2.currentText() == "PCA"):
            X=self.XPCA
        elif(self.comboBox_2.currentText() == "chi2"):
            X=self.Xchi2
        elif(self.comboBox_2.currentText() == "No Selection"):
            X = self.data.iloc[:,0:4] 
            Y = self.data.iloc[:,-1]
            
        try:
            self.data.drop("Id",axis=1,inplace=True)
            X = self.data.iloc[:,0:4] 
            y = self.data.iloc[:,-1]       
            self.xtraincek(X_train)
            self.ytraintb.setText(str(Y_train.to_string()))
            self.xtestcek(X_test)
            self.ytesttb.setText(str(Y_test.to_string()))
            tree=DecisionTreeClassifier()
            tree.fit(X_train,Y_train)
            prediction=tree.predict(X_test)
            self.textEdit_cm.setText(str(confusion_matrix(Y_test, prediction)))
            self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
            acc = metrics.accuracy_score(prediction,Y_test)
            self.acc.setText("ACC : {:.2f}%".format(acc))
            self.pltRoc(Y_test,prediction,"DTROC")
            self.confmat(Y_test, prediction, "DT")
        except:
            self.xtraincek(X_train)
            self.ytraintb.setText(str(Y_train.to_string()))
            self.xtestcek(X_test)
            self.ytesttb.setText(str(Y_test.to_string()))
            tree=DecisionTreeClassifier()
            tree.fit(X_train,Y_train)
            prediction=tree.predict(X_test)
            self.textEdit_cm.setText(str(confusion_matrix(Y_test, prediction)))
            self.label_8.setText('Train Sayısı : {} Test Sayısı: {}'.format(X_train.shape[0], X_test.shape[0]))
            acc = metrics.accuracy_score(prediction,Y_test)
            self.acc.setText("ACC : {:.2f}%".format(acc))
            self.pltRoc(Y_test,prediction,"DTROC")
            self.confmat(Y_test, prediction, "DT")

    def loggrid(self):
        
        # normalization
        X = self.data.iloc[:,0:4] 
        y = self.data.iloc[:,-1]
        X=(X-np.min(X))/(np.max(X)-np.min(X))
        x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.3)
        grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
        logreg=LogisticRegression()
        logreg_cv=GridSearchCV(logreg,grid,cv=10)
        logreg_cv.fit(x_train,y_train)
        print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
        print("accuracy :",logreg_cv.best_score_)
        logreg2=LogisticRegression(C=1,penalty="l2")
        logreg2.fit(x_train,y_train)
        print("score",logreg2.score(x_test,y_test))
     


    def bagging(self):
        X = self.data.iloc[:,0:4] 
        y = self.data.iloc[:,-1]
        X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)
        X, y = make_classification(n_samples=100, n_features=4,
                                   n_informative=2, n_redundant=0,
                                   random_state=0, shuffle=False)
        clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0).fit(X_train, Y_train)
        clf.predict(X_test)
        prediction=clf.predict(X_test)
        self.textEdit_4.setText(str(clf.predict(X_test)))
        self.textEdit_3.setText(str(clf.score(X_test,Y_test)))
        self.pltRoc(Y_test,prediction,"bagging")
        self.confmat(Y_test, prediction, "bagging")
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid green;}")


    def voting(self):
        X = self.data.iloc[:,0:4] 
        y = self.data.iloc[:,-1]
        X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)
        estimator = [] 
        estimator.append(('LR',  
                  LogisticRegression(solver ='lbfgs',  
                                     multi_class ='multinomial',  
                                     max_iter = 200))) 
        estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
        estimator.append(('DTC', DecisionTreeClassifier())) 
        vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 
        vot_hard.fit(X_train, Y_train) 
        y_pred = vot_hard.predict(X_test)
        score = accuracy_score(Y_test, y_pred) 
        self.textEdit_3.setText("Hard Voting Score : " + str(score))
        vot_soft = VotingClassifier(estimators = estimator, voting ='soft') 
        vot_soft.fit(X_train, Y_train) 
        y_pred = vot_soft.predict(X_test)
        score = accuracy_score(Y_test, y_pred)
        self.textEdit_3.setText(self.textEdit_3.toPlainText() + "\n" + "Soft Voting Score : " + str(score))
        self.pltRoc(Y_test,y_pred,"voting")
        self.confmat(Y_test, y_pred, "voting")
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid green;}") 



    def boosting(self):
        X = self.data.iloc[:,0:4] 
        y = self.data.iloc[:,-1]
        clf = AdaBoostClassifier(n_estimators=100)
        scores = cross_val_score(clf, X, y, cv=5)
        self.textEdit_3.setText(str(scores))





    def rocegr(self,y_test,y_pred,baslik):
        lr_auc = roc_auc_score(y_test, y_pred)  
        print('ALGRTM: ROC AUC=%.3f' % (lr_auc))
        lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label=baslik)
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
#        pyplot.savefig('RocCnn.png')
        pyplot.legend()
        pyplot.show()     
#        photo_path2 = "./RocCnn.png"
#        self.label_25.setPixmap(QPixmap(photo_path2))


    def confmat(self,y_test,y_pred,isim):
        cm = confusion_matrix(y_test, y_pred)
        cm_data = pd.DataFrame(cm)
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        sensitivity = (tp/(tp+fn))
        specificity = (tn/(tn+fp))
        plt.figure(figsize = (5,5))
        sns.heatmap(cm_data, annot=True,fmt="d")
        plt.title(isim)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        pathlib.Path('./PCAs').mkdir(parents=True, exist_ok=True)
        plt.savefig("./cnfmat.png")
        self.pixmap = QPixmap("./cnfmat.png")
        self.label_29.setPixmap(self.pixmap)
        self.textEdit_6.setText(str(sensitivity))
        self.textEdit_7.setText(str(specificity))
        plt.show()





    def pltRoc(self,y_test,y_pred,baslik):    
        from sklearn import preprocessing
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics
        from collections import Counter
        le = preprocessing.LabelEncoder()      
        y_test = le.fit_transform(y_test)
        y_pred = le.fit_transform(y_pred)
        y_test=np.array(y_test)
        y_pred=np.array(y_pred)
        postotal=0
        for i in range(2):
            if np.count_nonzero(y_pred == i)!=0:
                postotal+=1
        postotal1=0
        for i in range(2):
            if np.count_nonzero(y_test == i)!=0:
                postotal1+=1
                
        if postotal==postotal1:
            lr_fpr, lr_tpr, thresholds  =metrics.roc_curve(y_test, y_pred, pos_label=postotal)
            plt.plot(lr_fpr, lr_tpr, marker='.', label='baslik')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig('roc.png')
            plt.show()   
            photo_path2 = "./roc.png"
            self.label_9.setPixmap(QPixmap(photo_path2))
        else:
            lr_fpr, lr_tpr, thresholds  =metrics.roc_curve(y_test, y_pred, pos_label=postotal1)
            plt.plot(lr_fpr, lr_tpr, marker='.', label='baslik')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig('roc.png')
            plt.show()   
            photo_path2 = "./roc.png"
            self.label_9.setPixmap(QPixmap(photo_path2))
            
       
    
    def cnn(self):
        X = self.data.iloc[:,0:4] 
        y = self.data.iloc[:,-1]
        le = preprocessing.LabelEncoder()      
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        y_test = le.fit_transform(y_test)
        y_train= le.fit_transform(y_train)
        # y_pred = le.fit_transform(y_pred)
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(X.shape[1]))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            batch_size=int(self.bstb_4.text()),
            epochs=int(self.eptb_3.text()))
      
        
        y_true = np.array(y_test)
        y_pred = np.squeeze(np.array(model.predict(X_test) ,dtype=np.int))
        self.textEdit_8.setText(str(history.history['accuracy']) + "\n" + str(history.history['loss']))
        self.textEdit_9.setText(str(round(accuracy_score(y_true, y_pred)*100, 2)))
        cm = confusion_matrix(y_true, y_pred)
        self.textEdit_5.setText(str(cm))
        self.pltRoc(y_test,y_true,"CNN ROC")
        self.confmat(y_test, y_pred, "CNN CNF")
        self.groupBox_5.setStyleSheet("QGroupBox { border: 1px solid green;}")
        print("Tamamlandı.")
        
    
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec())
    
    
   