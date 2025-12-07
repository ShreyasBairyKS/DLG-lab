import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
import numpy as np

num_words = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=128, input_length=maxlen))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 2])
    return decoded_review

def predicted_sentiment(review):
    word_index = imdb.get_word_index()
    encoded_preview = [word_index.get(word, 0) + 3 for word in review.split()]
    padded_review = pad_sequences([encoded_preview], maxlen=maxlen)
    prediction = model.predict(padded_review, verbose=0)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment

for i in range(5):
    review = decode_review(X_test[i])
    sentiment = predicted_sentiment(review)
    print(f'Review: {review}\nPredicted Sentiment: {sentiment}\n')