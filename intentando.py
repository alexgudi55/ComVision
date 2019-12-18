import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from math import ceil

BATCH_SIZE = 32
DROPOUT_PROB = 0.2
imaGen = ImageDataGenerator(validation_split = 0.2)

training_set = imaGen.flow_from_directory(directory='slices',target_size=(80,80), subset="training", batch_size=32)
validation_set = imaGen.flow_from_directory(directory='slices',target_size=(80,80), subset="validation", batch_size=32)

#Load the VGG model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

for layer in base_model.layers:
    layer.trainable = False

print ("Base Model summary")
print(base_model.summary())

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_PROB)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(DROPOUT_PROB)(x)
predictions = Dense( 3,activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print ("Final Model summary")
model.summary()


# Fit model
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(
    generator= training_set,
    steps_per_epoch=ceil(training_set.samples / BATCH_SIZE),

    validation_data=validation_set,
    validation_steps=ceil(validation_set.samples / BATCH_SIZE),
    
    epochs=15,
    verbose=1,
)

import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

model.save('weights/image_model.h5')