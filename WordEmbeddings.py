
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, GlobalMaxPooling1D, Activation
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras

from keras.preprocessing.sequence import pad_sequences

# Loading data

dataset = pd.read_csv("full_amazon_ff_reviews.csv")
# counts null values
print(dataset.isnull().any().sum())
# remove null values
dataset = dataset.dropna(how='any',axis=0)


# x & y holds values
X = dataset['Text']
y = dataset['Rating']


# create training and test sets with test as 28%
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.28, random_state=53)

# (train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=20000)

print("Training data: {}, labels: {}".format(len(train_data), len(train_labels)))
# Pre-processing
num_words = 5000
max_len = 400
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 5
batch_size = 32

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data)
train_data = tokenizer.texts_to_sequences(train_data)
test_data = tokenizer.texts_to_sequences(test_data)

train_data = pad_sequences(train_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)
print(train_data)

# integer encoding
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.fit_transform(test_labels)
print(train_labels)

# one hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
train_labels = train_labels.reshape(len(train_labels), 1)
test_labels = test_labels.reshape(len(test_labels), 1)
onehot_encoded_train = onehot_encoder.fit_transform(train_labels)
onehot_encoded_test = onehot_encoder.fit_transform(test_labels)
print(train_labels)

#train_labels = to_categorical(encoded)
#test_labels = to_categorical(encoded))

# creating model
model = Sequential()
model.add(Embedding(num_words, embedding_dims, input_length=max_len))

# prevents overfitting
model.add(Dropout(0.5))
model.add(Conv1D(filters, kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(filters, kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_data, test_labels))

loss, accuracy = model.evaluate(train_data, train_labels, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))