# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:16:32 2020

@author: ARDI
"""

import training
import os
import numpy as np
import logging
from tkinter import Tk
from tkinter.filedialog import askopenfilename

logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('training').setLevel(logging.CRITICAL)
model_dir = 'model/model.h5'   

def start_test():
    Tk().withdraw()
    filename = askopenfilename(filetypes=[("Audio files (.wav)", "*.wav")])
    if filename != '':
        print(filename)
        model = training.create_model(summary=False)
        model.load_weights(model_dir)
        fr, sound = training.read(filename)
        classes = model.predict(np.vstack([sound[:10000]]), batch_size=10)[0][0]
        print(classes)
        if classes == 0:
            print('male')
        else:
            print('female')
    else:
        print('Tidak ada file yang dipilih')
        
if (os.path.isfile(model_dir)):
    start_test()
else:
    training.start_train()
    start_test()