import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils



images = []
classNum = []


path = 'archive/train'


myList = os.listdir(path)
numOfClasses = len(myList)

for i in range(0,numOfClasses):
    myPicList = os.listdir(path+"/"+myList[i])
    for j in myPicList:
        curImg = cv.cv2.imread(path+"/"+myList[i]+"/"+j)
        curImg = cv.cv2.resize(curImg,(120,120))
        images.append(curImg)
        classNum.append(i)
    print(myList[i],end=" ")
print(" ")

images = np.array(images)
classNum = np.array(classNum)

#print(images.shape)
#print(classNum.shape)

x_train,x_test,y_train,y_test = train_test_split(images,classNum,test_size = 0.2)


x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#print(x_train.shape)
#print(x_test.shape)

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape = (120,120,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(13, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(x_train.shape)
#print(y_train.shape)

history = model.fit(x_train,y_train, epochs=25)

val_loss,val_acc = model.evaluate(x_test,y_test) 
print("loss-> ",val_loss,"\nacc-> ",val_acc)

#Running this code will create a new model, to use the new model change its name in HandWrittenEquationSolver.py line : 88
model.save("Model_2.model")

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()