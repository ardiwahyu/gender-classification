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
        female_prob = model.predict(np.vstack([sound[:10000]]), batch_size=10)[0][0]
        male_prob = 1 - female_prob
        gender = "male" if male_prob > female_prob else "female"
        print("Result: ", gender)
        print(f"Probabilities: Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
    else:
        print('Tidak ada file yang dipilih')
        
if (os.path.isfile(model_dir)):
    start_test()
else:
    training.start_train(False)
    start_test()