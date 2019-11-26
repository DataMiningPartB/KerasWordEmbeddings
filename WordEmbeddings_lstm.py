import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, GlobalMaxPooling1D, Activation, \
    GlobalMaxPool1D, ConvLSTM2D, BatchNormalization, Conv3D
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from hyperas.distributions import uniform
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
sum = 0
for item in X:
    sum = sum + len(item.split())
print(sum)
# create training and test sets with test as 28%
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.33, random_state=42)

# (train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=20000)

print("Training data: {}, labels: {}".format(len(train_data), len(train_labels)))
# Pre-processing
num_words = 10000
max_len = 10000
embedding_dims = 50
filters = 16
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
# One-hot encoding needs to be done
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.fit_transform(test_labels)

print(train_labels)
train_labels = to_categorical(train_labels, dtype='float32')
test_labels = to_categorical(test_labels, dtype='float32')

# creating model
model = Sequential()
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])


model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_data, test_labels))


loss, accuracy = model.evaluate(train_data, train_labels, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))