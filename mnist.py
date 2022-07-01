#!/usr/bin/env python
# coding: utf-8

# In[32]:


import tensorflow as tf
import keras
from tensorflow.keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dense, Dropout, Flatten
# 定数の定義
img_rows, img_cols = 28, 28 
input_shape = (img_rows, img_cols, 1) 
num_classes = 10

# モデルの構築
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


# In[33]:


model.summary()


# In[34]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.gradient_descent_v2.SGD(),
              metrics=['accuracy'])


# In[36]:


# MNISTデータをロード
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[37]:


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
x_test = x_test.astype('float32')
x_test /= 255


# In[38]:


y_train = np_utils.to_categorical(y_train, num_classes) 
y_test = np_utils.to_categorical(y_test, num_classes)


# In[39]:


print('x_train.shape:', x_train.shape) 
print('x_test.shape:', x_test.shape) 
print('y_train.shape:', y_train.shape) 
print('y_test.shape:', y_test.shape)


# In[40]:


model.fit(x_train, 
          y_train, 
          batch_size=128,
          epochs=12,
          verbose=1)


# In[41]:


score = model.evaluate(x_test, y_test, verbose=0)


# In[42]:


print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[43]:


model.save('./KerasMnist.h5')

