# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.utils import np_utils
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
(x_train,y_train), (x_test, y_test)= cifar10.load_data()

batch_size = 32
num_classes = 10
epochs = 1

for i in range(6):
    plt.subplot(331+i)
    random_index = np.random.randint(0,len(x_train))
    plt.imshow(x_train[random_index])

x_train = x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



model = load_model("cifar10AnimalModel.h5")
# scores = model.evaluate(x_test, y_test,verbose=1)
result = model.predict_classes(x_test[101].reshape(1,32,32,3),1,verbose=1)[0]
print(result)
cv2.imshow("image",x_test[101])
cv2.waitKey(0)
cv2.destroyAllWindows()