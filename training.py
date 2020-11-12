# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:47:40 2020

@author: ARDI
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import pydub
import numpy as np
import os

def read(f, normalized=False):
    a = pydub.AudioSegment.from_wav(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1,2))
    if normalized:
        return a.frame_rate, np.float32(y)/2**15
    else:
        return a.frame_rate, y
    
def load_data(length=10000):  
    base_dir = '/Users/ARDI/Documents/Python Scripts/ML/data/gender'
    
    train_dir = os.path.join(base_dir, 'train_data')
    val_dir = os.path.join(base_dir, 'val_data')
    
    male_train_dir = os.path.join(train_dir, 'male')
    female_train_dir = os.path.join(train_dir, 'female')
    
    male_val_dir = os.path.join(val_dir, 'male')
    female_val_dir = os.path.join(val_dir, 'female')
    
    male_train = [os.path.join(male_train_dir, f) for f in os.listdir(male_train_dir)]
    female_train = [os.path.join(female_train_dir, f) for f in os.listdir(female_train_dir)]
    male_val = [os.path.join(male_val_dir, f) for f in os.listdir(male_val_dir)]
    female_val = [os.path.join(female_val_dir, f) for f in os.listdir(female_val_dir)]

    x_train = np.zeros((len(male_train)+len(female_train), length))
    y_train = np.zeros((len(male_train)+len(female_train), 1))
    for i, f in enumerate(male_train):
        fr, amp = read(f)
        x_train[i] = amp[:length]
        y_train[i] = 0
    for i, f in enumerate(female_train):
        fr, amp = read(f)
        x_train[i+len(male_train)] = amp[:length]
        y_train[i+len(male_train)] = 1
        
    x_val = np.zeros((len(male_val)+len(female_val), length))
    y_val = np.zeros((len(male_val)+len(female_val), 1))
    for i, f in enumerate(male_val):
        fr, amp = read(f)
        x_val[i] = amp[:length]
        y_val[i] = 0
    for i, f in enumerate(female_val):
        fr, amp = read(f)
        x_val[i+len(male_val)] = amp[:length]
        y_val[i+len(male_val)] = 1
        
    return x_train, y_train, x_val, y_val
    
def create_model(summary=True, length=10000):
    model = Sequential()
    model.add(Dense(256, input_shape=(length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    if summary:
        model.summary()
    return model

def start_train():
    model = create_model()
    
    tensorboard = TensorBoard(log_dir="logs")
    early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)
    x_train, y_train, x_val, y_val = load_data()
    model.fit(x_train, y_train, epochs=20, steps_per_epoch=100,
              validation_data=(x_val, y_val),
              verbose=2, callbacks=[tensorboard, early_stopping])
    model.save("model/model.h5")