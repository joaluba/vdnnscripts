#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import random 
import numpy as np
from matplotlib import pyplot as p 
from scipy.io import loadmat,savemat 

import hdf5storage

'TRAINING SET'
# load train set from .mat files
y_trainbig=hdf5storage.loadmat('../data/test_x_big.mat')
y_trainbig=y_trainbig['g']
y_train=y_trainbig[:,50000:300000]
del y_trainbig
x_trainbig=hdf5storage.loadmat('../data/test_y_big.mat')
x_trainbig=x_trainbig['s']
x_train=x_trainbig[0:3,50000:300000]
del x_trainbig

'shuffle data in the training set'
indi = np.arange(y_train.shape[1])
np.random.shuffle(indi)
y_train=y_train[:,indi]
x_train=x_train[:,indi]

'TEST SET'
# load test set from .mat files
x_test=hdf5storage.loadmat('../data/test_x.mat')
x_test=x_test['g']
y_test=hdf5storage.loadmat('../data/test_y.mat')
y_test=y_test['s']
y_test=y_test[0:3,:]


'Normalising the data: test data is scaled the same as train data'
scaler_x=StandardScaler()
x_train=np.transpose(scaler_x.fit_transform(np.transpose(x_train)))
x_test=np.transpose(scaler_x.fit_transform(np.transpose(x_test)))
scaler_y=StandardScaler()
y_train=np.transpose(scaler_y.fit_transform(np.transpose(y_train)))

'split the training set into training and evaluation'
train_In=x_train[:,50000:]
train_Tar=y_train[:,50000:]
eval_In=x_train[:,:50000]
eval_Tar=y_train[:,:50000]

''' Import everything we need for DNN-Construction '''
#Load Keras Model Toolbox
from keras.models import Sequential 
from keras.layers import Dense 

'DNN REGRESSION NETWORK'
model = Sequential()
# input layer: 3 dimensional voice state: F0,F1,F2 
model.add(Dense(units=100,activation='sigmoid',input_dim=3)) 
model.add(Dense(1000,activation='sigmoid')) 
model.add(Dense(2576,activation='sigmoid')) 
model.compile('Nadam','logcosh') 
model.summary()
model.save('../results/revmodel_vDNNb.h5')

'TRAINING THE NETWORK'
training=model.fit(np.transpose(train_In),np.transpose(train_Tar),epochs=50,validation_data=(np.transpose(eval_In),np.transpose(eval_Tar))) 
histod=training.history
lossvals=histod['loss']
vallossvals=histod['val_loss']
plt.figure()
plt.plot(lossvals,'bo',label='training loss')
plt.xlabel('EPOCHS')
plt.ylabel('TRAINING LOSS')
plt.title('training set with 50000 d.p.')
plt.show()
plt.savefig('trainingloss_exp1.png')
plt.plot(lossvals,'ro',label='validation loss')


'TESTING THE NETWORK'
x_test=np.transpose(x_test)
y_predi_sc=model.predict(x_test)
y_predi=scaler_y.inverse_transform(y_predi_sc)
y_test=np.transpose(y_test)

for i in range(0, 1000):
    print("true value:", y_test[i,:])
    print("estimated value:", y_predi[i,:])
    
pred_RMSE_F0=np.sqrt(metrics.mean_squared_error(y_predi,y_test))
print("Final RMSE O:{}".format(pred_RMSE_F0))

'SAVE RESULTS'
#savemat('results/y_predi_vDNNb.mat', mdict={'y_predi': y_predi})
#savemat('results/y_test_vDNNb.mat', mdict={'y_test': y_predi})

