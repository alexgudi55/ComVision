import numpy as np
import pandas as pd
import os
import keras
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from math import ceil
from keras.preprocessing import image

img = image.load_img("image.jpg",target_size=(80,80))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("weights/image_model.h5")
output = saved_model.predict(img)

print(output)