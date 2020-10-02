#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:23:33 2020
@author: joanna luberadzka
"""
from keras.models import load_model
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def predictgl(s,s_model,name_in_scaler,name_out_scaler):
    #s=np.array([F0, F1, F2])
    model = load_model(s_model)
    in_scaler = joblib.load(name_in_scaler) 
    out_scaler = joblib.load(name_out_scaler) 
    s_sc=in_scaler.transform(s.reshape(1,-1))
    glimpse_sc=model.predict(s_sc)
    glimpse=out_scaler.inverse_transform(glimpse_sc)
    return glimpse

#s=np.array([100, 350, 2400])
#a=predictgl(100,400,2500,"my_modelvoiceDNN_rev.h5","scaler_x_reva.save","scaler_y_reva.save")