#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:20:33 2017

@author: wq
"""

import time
from ovito.data import *
from sklearn.externals import joblib
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
dis = 0.4
width = 10
hight = 10
length = 10

def to_color(y):
    if y == 1:
        return (0, 0.9, 0)
    elif y == 2:
        return (0.9, 0, 0)
    elif y == 3:
        return (0, 0, 1)
    else:
        return (1, 1, 1)

def get_input_data(data, select):
	
    pos = data.particle_properties.position.array
    s = time.clock()
    x = pos.T[0].copy() + 10 # - min(pos.T[0])
    y = pos.T[1].copy() + 10#- min(pos.T[1])
    z = pos.T[2].copy() + 20#- min(pos.T[2])

    x /= dis
    y /= dis
    z /= dis
    
    x = x.astype(int)
    y = y.astype(int)
    z = z.astype(int)
   
    max_x = max(x) + width
    max_y = max(y) + hight
    max_z = max(z) + length

    buf_cube = np.zeros(shape = (max_x, max_y, max_z), dtype = bool)
    
    for i in range(len(x)):
        buf_cube[x[i], y[i], z[i]] = 1

    e = time.clock()
    print("create cube spend %f second"%(e - s))
    s = time.clock()

    size = 8* width * length * hight
    l = len(x)
    fun = lambda i: np.array(buf_cube[(x[i] - width):(x[i] + width), (y[i] - hight):(y[i] + hight), (z[i] - length):(z[i] + length)])
    feature = np.array(list(map(fun, range(l)))).reshape((l, size)) 

    e = time.clock()
    print("each feature select spend %f second"%(e - s))
	
    feature = feature[:, select]
    e = time.clock()
    print("feature select spend %f second	"%(e - s))
    return feature
    
	
def modify(frame, input, output):
    
    s = time.clock()
    f = open("feature_keras.txt", "r")
    select = f.readlines()[0]
    select = list(map(int, select.split(" ")))
    #MLP = joblib.load("/home/wq/soft/ovito-2.9.0-x86_64/MLP.model")
    e = time.clock()
    print("MLP %f second"%(e - s))
    
    data = get_input_data(input, select)
    e = time.clock()
    print("feature %f second"%(e - s))
    
    MLP = Sequential()
    MLP.add(Dense(20, activation='relu', input_dim=1000))
    MLP.add(Dropout(0.5))
    MLP.add(Dense(32, activation='relu'))
    MLP.add(Dropout(0.5))
    MLP.add(Dense(64, activation='relu'))
    MLP.add(Dropout(0.5))
    MLP.add(Dense(64, activation='relu'))
    MLP.add(Dropout(0.5))
    MLP.add(Dense(128, activation='softmax'))
    MLP.add(Dropout(0.5))
    MLP.add(Dense(3, activation='softmax'))
    
    MLP.load_weights('keras_nn.h5')
    
    
    pred = MLP.predict(data)
    e = time.clock()
    print("pred %f second"%(e - s))
    temp = [np.argmax(i) for i in pred]
    for i in set(temp):
        print(i, temp.count(i))
    color_property = output.create_particle_property(ParticleProperty.Type.Color)
    color_property.marray[:]=[(i[1], i[0], i[2]) for i in pred]
    print("24The input contains %i particles." % input.number_of_particles)