from keras.models import Sequential
from keras.layers import Dense
import numpy as np
input=np.array([[0,0],[0,1],[1,0],[1,1]])
output=np.array([[0],[1],[1],[0]])

model=Sequential([
    Dense(8,activation='relu',input_shape=(2,)),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(input,output,epochs=1000)
_,acc=model.evaluate(input,output)

print("The model accuracy is: ",f"{acc:.4f}")