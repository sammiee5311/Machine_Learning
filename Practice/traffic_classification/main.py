import os
import cv2
import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

path = './dataset'
images = []
datasets = []
labels = []
dataframe = pd.read_csv('./labels/labels.csv')

n_classes = os.listdir(path)
print('# of classes :', len(n_classes))
print('processing :')
for lists in n_classes:
    print(lists, end=' ')
    image_path = os.listdir(path+'/'+lists)
    for images in image_path:
        img = cv2.imread(path+'/'+lists+'/'+images)
        img = cv2.resize(img, (32,32))
        datasets.append(img)
        labels.append(lists)

datasets = np.asarray(datasets)
labels = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
)

test_datagen = ImageDataGenerator()


y_train = to_categorical(y_train,len(n_classes))
y_test = to_categorical(y_test, len(n_classes))
y_validation = to_categorical(y_validation, len(n_classes))

model = models.Sequential()
model.add(layers.Conv2D(60, (5,5), input_shape=(32, 32, 1), activation='relu'))
model.add(layers.Conv2D(60, (5,5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(30, (3, 3), activation='relu'))
model.add(layers.Conv2D(30, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(units=500, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=len(n_classes), activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

print(model.summary())

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'my_log_dir',
        histogram_freq=1,
        embeddings_freq=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=2
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=64
)

validation_generator = test_datagen.flow(
    X_validation,
    y_validation
)

model.fit_generator(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks,
    shuffle=1
)
