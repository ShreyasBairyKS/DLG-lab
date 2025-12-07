from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt

def create_data():
    x=np.random.randn(1000,10)
    y=np.random.randn(1000,1)
    return x,y

def create_model():
    model=Sequential([
        Dense(50,activation='relu',input_shape=(10,)),
        Dense(20,activation='relu'),
        Dense(1)
    ])
    return model

def run_prog(model,optimizer,x,y,optimizer_name,epoch,batch_size):
    model.compile(loss='mse',optimizer=optimizer)
    history=[]
    print(optimizer_name)
    for i in range(epoch):
        hist=model.fit(x,y,batch_size=batch_size,epoch=1)
        loss=hist.history['loss'][0]
        print(f"{i+1} epoch/{epoch} and loss= {loss:.4f}")
        history.append(loss)
    return history

X,Y=create_data()
ad=create_model()
sgd=create_model()

his_adam=run_prog(ad,Adam(learning_rate=0.001),X,Y,"Adam",50,32)
his_sgd=run_prog(sgd,SGD(learning_rate=0.01),X,Y,50,32)

plt.plot(range(1,50+1),his_adam,label="Adam",color='red')
plt.plot(range(1,50+1),his_sgd,label="SGD",color='blue')
plt.title("SGD vs ADAM")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.grid(True)
plt.show()