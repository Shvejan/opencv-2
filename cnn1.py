from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD 

(x_train,y_train),(x_test, y_test) = mnist.load_data()


# for i in range(0,6):
#     plt.subplot(331+i)
#     random_num = np.random.randint(0,len(x_train))
#     plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))



x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_train=x_train.astype('float32')
x_test = x_test.astype('float32')

#normalizing
x_train /=255
x_test /=255

#hot one encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#creating the model
input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1)
num_classes = y_test.shape[1]

model = Sequential()


model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.01),
              metrics = ['accuracy'])



#training the model
# batch_size = 32
# epochs = 1

# history = model.fit(x_train,
#                     y_train,
#                     batch_size = batch_size,
#                     epochs = epochs,
#                     verbose = 1,
#                     validation_data = (x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)

classifier = load_model('mnistNumbers.h5')
score = classifier.evaluate(x_test, y_test)
print(score)

for i in range(8):
    plt.subplot(331+i)
    random_num = np.random.randint(0,len(x_test))
    result = classifier.predict_classes(x_test[random_num].reshape(1,28,28,1),1,verbose=0)
    print(result[0])
    plt.imshow(x_test[random_num])
    

