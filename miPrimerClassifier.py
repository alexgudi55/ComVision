import numpy as np
import copy
import pandas as pd
import os
import cv2
"""import keras
from matplotlib import pyplot as plt

from keras.models import Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from math import ceil
"""
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 5
DROPOUT_PROB = 0.2
DATASET_PATH = "formated/"



#print("DATASET_PATH content")
#print(os.listdir(DATASET_PATH))
list = {'Image' : os.listdir(DATASET_PATH)}
# Read CSV file
df = pd.read_csv("trainNodules_gt.csv", nrows=None, error_bad_lines=False)
df1 = pd.DataFrame(list)

iden = []
flag = False
for id in df1['Image']:
    s = str(    )
    for c in id:
        if c == 'd':
            flag = True
            continue
        if c == 'R':
            flag = False
            break    
        if flag:
            s = s + c
    iden.append(s)

lista = {'id' : iden}
df2 = pd.DataFrame(lista)
df1 = pd.concat([df1, df2], axis = 1)
#print(df1)

texture = []
for index, id in  enumerate(df1['id']):
    for ind, iden in enumerate(df['LNDbID']):
        print(id,iden)
        if id == iden:
            texture.append(df['Text'][ind])
