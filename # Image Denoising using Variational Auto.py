import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)
print(x_test.shape)

num_pixels=x_train.shape[1]*x_train.shape[2]
x_train=x_train.reshape(-1,num_pixels).astype('float32')/255
x_test=x_test.reshape(-1,num_pixels).astype('float32')/255

x_train_noisy=np.clip((x_train+0.2*np.random.normal(0,1,x_train.shape)),0.,1.)
x_test_noisy=np.clip((x_test+0.2*np.random.normal(0,1,x_test.shape)),0.,1.)

model=Sequential([
    Dense(500,activation='relu',input_dim=num_pixels),
    Dense(300,activation='relu'),
    Dense(100,activation='relu'),
    Dense(300,activation='relu'),
    Dense(500,activation='relu'),
    Dense(784,activation='sigmoid')
])

model.compile(loss='mse',optimizer='Adam')

model.fit(x_train_noisy,x_train,validation_data=(x_test_noisy,x_test),batch_size=200,epochs=2)
pred=model.predict(x_test_noisy)

x_test_noisy=x_test_noisy.reshape(-1,28,28)*255
x_test=x_test.reshape(-1,28,28)*255
pred=pred.reshape(-1,28,28)*255

plt.figure(figsize=(20,4))
for i in range(10,20,1):
    plt.subplot(2,10,i+1)
    plt.imshow(x_test[i,:,:],cmap='grey')
    plt.title(y_test[i])
plt.show()

plt.figure(figsize=(20,4))
for i in range(10,20,1):
    plt.subplot(2,10,i)
    plt.imshow(x_test_noisy[i,:,:],cmap='grey')
    plt.title(y_test[i])
plt.show()

plt.figure(figsize=(20,4))
for i in range(10,20,1):
    plt.subplot(2,10,i+1)
    plt.imshow(pred[i,:,:],cmap='grey')
    plt.title(y_test[i])
plt.show()