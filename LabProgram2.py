import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1), metrics=['accuracy'])

model.fit(X, y, epochs=10000, verbose=0)

_, accuracy = model.evaluate(X, y)
print(f'Accuracy: {accuracy * 100:.2f}%')

predictions = model.predict(X)
print(np.round(predictions))
