from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train),(x_test, y_test) = mnist.load_data()


# for i in range(0,6):
#     plt.subplot(331+i)
#     random_num = np.random.randint(0,len(x_train))
#     plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))



x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_train=x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /=255
x_test /=255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


