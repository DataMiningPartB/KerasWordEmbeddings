import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from hyperopt import tpe, Trials
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, GlobalMaxPooling1D, Activation, \
    GlobalMaxPool1D
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
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
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.33, random_state=42)

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_data)
    train_data = tokenizer.texts_to_sequences(train_data)
    test_data = tokenizer.texts_to_sequences(test_data)
    print(len(tokenizer.word_counts))
    train_data = pad_sequences(train_data, maxlen=10000)
    test_data = pad_sequences(test_data, maxlen=10000)
    print(train_data)
    # One-hot encoding needs to be done
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.fit_transform(test_labels)

    print(train_labels)
    train_labels = to_categorical(train_labels, dtype='float32')
    test_labels = to_categorical(test_labels, dtype='float32')
    return train_data, train_labels, test_data, test_labels
    # (train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=20000)
    print("Training data: {}, labels: {}".format(len(train_data), len(train_labels)))
# Pre-processing


# creating model

def init_model(train_data, train_labels, test_data, test_labels):
    num_words = 10000
    max_len = 10000
    embedding_dims = 50
    filters = 16
    kernel_size = 3
    hidden_dims = 250
    epochs = 5
    batch_size = 32
    model = Sequential()
    # hyper paramaters
    model.add(Conv1D(hidden_dims, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(64,1)))
    model.add(GlobalMaxPooling1D())

    # prevents overfitting
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(250))

    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([3, 4, 5])}}))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    score = model.evaluate(test_data, test_labels, verbose=0)
    accuracy = score[1]
    return {'loss': -accuracy, 'model': model}

run, model = optim.minimize(model=init_model,
                            data=data,
                            algo=tpe.suggest,
                            max_evals=5,
                            trials=Trials())
train_data, train_labels, test_data, test_labels = data()
print("Best performing model:")
print(model.evaluate(test_data, test_labels))

print("Best performing model chosen by hyper-parameters:")
print(run)
# loss, accuracy = model.evaluate(train_data, train_labels, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(test_data, test_labels, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
