#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:13:31 2020

@author: medi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import random 

import numpy as np #Numpy is THE Math toolbox for Python

from matplotlib import pyplot as p #all plotting commands you are used to in matlab, bundled in one package!

from scipy.io import loadmat,savemat #Easy read and write of matlab .mat files. I usually do all my plotting stuff in Matlab - it's just more convenient =)

import hdf5storage

'TRAINING SET'
x_trainbig=hdf5storage.loadmat('../data/test_x_big.mat')
x_trainbig=x_trainbig['g']
x_train=x_trainbig[:,50000:300000]
del x_trainbig
y_trainbig=hdf5storage.loadmat('../data/test_y_big.mat')
y_trainbig=y_trainbig['s']
y_train=y_trainbig[0:3,50000:300000]
del y_trainbig
'shuffle the data in the array'
indi = np.arange(y_train.shape[1])
np.random.shuffle(indi)
y_train=y_train[:,indi]
x_train=x_train[:,indi]
'TEST SET'
x_test=hdf5storage.loadmat('../data/5t_test_x.mat')
x_test=x_test['g']
y_test=hdf5storage.loadmat('../data/5t_test_y.mat')
y_test=y_test['s']
y_test=y_test[0:3,:]


'Normalising the data: test data is scaled the same as train data'
scaler_x=StandardScaler()
x_train=np.transpose(scaler_x.fit_transform(np.transpose(x_train)))
x_test=np.transpose(scaler_x.fit_transform(np.transpose(x_test)))
scaler_y=StandardScaler()
y_train=np.transpose(scaler_y.fit_transform(np.transpose(y_train)))
from keras.models import load_model
model = load_model('../results/model_vDNNb.h5')
 
x_test=np.transpose(x_test)
y_predi_sc=model.predict(x_test)
y_predi=scaler_y.inverse_transform(y_predi_sc)
y_test=np.transpose(y_test)

for i in range(0, 1000):
    print("true value:", y_test[i,:])
    print("estimated value:", y_predi[i,:])
    
pred_RMSE_F0=np.sqrt(metrics.mean_squared_error(y_predi[:,0],y_test[:,0]))
pred_RMSE_F1=np.sqrt(metrics.mean_squared_error(y_predi[:,1],y_test[:,1]))
pred_RMSE_F2=np.sqrt(metrics.mean_squared_error(y_predi[:,2],y_test[:,2]))
print("Final RMSE F0:{}".format(pred_RMSE_F0))
print("Final RMSE F1:{}".format(pred_RMSE_F1))
print("Final RMSE F2:{}".format(pred_RMSE_F2))

plt.figure;
plt.plot(y_predi[0:1000,0])
plt.plot(y_test[0:1000,0])
savemat('../results/y_predi_paper.mat', mdict={'y_predi': y_predi})
savemat('../results/y_test_paper.mat', mdict={'y_test': y_test})




