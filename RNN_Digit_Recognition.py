import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,28,28)/255
x_test=x_test.reshape(-1,28,28)/255

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

model=Sequential([
    LSTM(128,activation='relu',input_shape=(28,28)),
    Dense(10,activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

test_loss,test_acc=model.evaluate(x_test,y_test)

print("Accuracy is: ",test_acc)