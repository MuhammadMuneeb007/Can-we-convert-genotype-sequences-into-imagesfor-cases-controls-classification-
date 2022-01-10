from __future__ import absolute_import, division, print_function
import argparse 
from tensorflow.keras import Model, layers
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from numpy import genfromtxt
from sklearn import svm 
from numpy import genfromtxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Conv1D, MaxPooling1D 
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
 
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
scaler = StandardScaler()
import tensorflow.keras
from sklearn.metrics import precision_score, recall_score, accuracy_score 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation,ActivityRegularization
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU,Bidirectional
from sklearn import preprocessing 
from tensorflow.keras.layers import Reshape 
import seaborn as sn
import matplotlib.pyplot as plt
from pylab import rcParams
import sys
import os
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def metric(name,best_model,x_train,y_train,x_test,y_test):
  
  y_pred = best_model.predict(x_test)
  sn.set(font_scale=2)
  rcParams['figure.figsize'] = 7, 7
  confusion_matrix = pd.crosstab(y_test.argmax(axis=1), y_pred.argmax(axis=1), rownames=['Actual'], colnames=['Predicted'])
  sn.heatmap(confusion_matrix, annot=True)

  plt.savefig(args.path+os.sep+name+"Test.png")
  plt.clf()
  confusion_matrix = pd.crosstab(y_train.argmax(axis=1), best_model.predict(x_train).argmax(axis=1), rownames=['Actual'], colnames=['Predicted'])
  sn.heatmap(confusion_matrix, annot=True)

  plt.savefig(args.path+os.sep+name+"Train.png")
  plt.clf()


def plotting(history):
  fig = plt.figure()
  history_dict = history.history
  print(history_dict.keys())
  plt.subplot(2,1,1)
  plt.plot(history_dict['accuracy'])
  plt.plot(history_dict['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Training Set', 'Validation Set'], loc='lower right')

  plt.subplot(2,1,2)


  plt.plot( history_dict['loss'])
  plt.plot( history_dict['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Training Set', 'Validation Set'], loc='upper right')

  plt.tight_layout()


import pandas as pd 
from os import makedirs

direc = sys.argv[1]  
trainpath= "./"+direc+os.sep+"train/"
testpath = "./"+direc+os.sep+"test/"
 
allsnps = os.listdir("./"+direc)
#from xgboost import XGBClassifier
for files in allsnps:
  #print(files)
  encoding = ["1"]
  activationFunctions = ["relu","sigmoid"]
  ddropout = [ 0.2]
  optimizer = ["Adam" ]
  batchsize = [50]
  epochsnumber = [50]
  validation = [0.3] 
  if "_snps" not in files and "pv_" in files:
    f = open(direc+os.sep+files+os.sep+"Result2.txt", "w")
    print(files)
    f.write(str("Snps"+","+str(files)))     
    for enc in encoding:
      if os.path.exists("./"+direc+os.sep+files+os.sep+'ptrain.raw'):
        x_train = pd.read_csv("./"+direc+os.sep+files+os.sep+'ptrain.raw', sep="\s+")
        x_test = pd.read_csv("./"+direc+os.sep+files+os.sep+'ptest.raw', sep="\s+") 
        y_train  = pd.read_csv(trainpath+'YRI.pheno', sep="\s+") 
        y_test= pd.read_csv(testpath+'YRI.pheno', sep="\s+")
        x_train =x_train.iloc[:,6:].values
        x_test  =x_test.iloc[:,6:].values
        y_train =y_train.iloc[:,2:].values
        y_test =y_test.iloc[:,2:].values
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
 
  
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
 
        x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.2)
        for activation in activationFunctions:
          for drop in ddropout:
            for opt in optimizer:
              for b in batchsize:
                for e in epochsnumber:
                
                  model = Sequential()
                  print(x_train.shape,x_test.shape,y_train.shape, y_test.shape)
                  model.add(Conv1D(32, 3, activation=activation, input_shape=(x_train.shape[1],1)))
                  #model.add(Conv2D(32, kernel_size=(3, 3),activation=activation,input_shape=(n_xtrain.shape[1],n_xtrain.shape[2],1)))
                  model.add(MaxPooling1D(2))
                  #model.add(Conv2D(64, (3, 3), activation=activation))
                  model.add(Conv1D(64, 3, activation=activation))
                  
                  model.add(MaxPooling1D(2))
                  model.add(Flatten())
                  model.add(Dense(10, activation=activation))
                  model.add(Dropout(drop))
                  model.add(Dense(2, activation='softmax'))
                  model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
                  history = model.fit(x_train, y_train,batch_size=b, epochs=e,validation_data=(x_val, y_val),verbose=0)
                  print(activation,drop,opt,b,e)
                  print("\n")
                  
                  f.write("\n")
                  f.write(str(str(activation)+","+str(drop)+","+str(opt)+","+str(b)+","+str(e)))
                  x,y = model.evaluate(x_train, y_train)
                  f.write("\n")
                  f.write(str("Training Accuracy"+str(y)))
                  f.write("\n")
                  x,y = model.evaluate(x_test, y_test)
                  f.write("Test Accuracy"+str(y))
                  f.write("\n")
                  x,y =model.evaluate(x_val, y_val)
                  f.write("Validation Accuracy"+str(y))
                  f.write("\n")
                  
                  print( model.evaluate(x_train, y_train))
                  print("\n")              
                  print( model.evaluate(x_test, y_test))
                  print("\n")
  
    f.close()




