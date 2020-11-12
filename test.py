# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:16:32 2020

@author: ARDI
"""

import training
import os
import numpy as np

model_dir = 'model/model.h5'
base_dir = '/Users/ARDI/Documents/Python Scripts/ML/data/gender'    

def start_test():
    model = training.create_model(summary=False)
    model.load_weights(model_dir)
    fr, sound = training.read(os.path.join(base_dir, 'fepi_0001.wav'))
    classes = model.predict(np.vstack([sound[:10000]]), batch_size=10)
    if classes == 0:
        print('male')
    else:
        print('female')
        
if (os.path.isfile(model_dir)):
    start_test()
else:
    training.start_train()
    start_test()