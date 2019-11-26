import pandas as pd
from hyperopt import tpe, Trials, STATUS_OK
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, Dense, GlobalMaxPooling1D, Activation, Flatten
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    # Loading data
    dataset = pd.read_csv("reduced_amazon_ff_reviews.csv")

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

    train_labels = to_categorical(train_labels, dtype='float32')
    test_labels = to_categorical(test_labels, dtype='float32')
    print(train_labels)

    # (train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words=20000)
    print("Training data: {}, labels: {}".format(len(train_data), len(train_labels)))

    return train_data, train_labels, test_data, test_labels
# Pre-processing


# creating model

def init_model(train_data, train_labels, test_data, test_labels):
    model = Sequential()

    model.add(Embedding({{choice([1000, 2500, 5000, 10000])}},
                        {{choice([25, 50, 100, 250])}},
                        input_length=10000))

    model.add(Conv1D({{choice([100, 250, 500, 1000])}},
                     {{choice([1, 2, 3, 4])}},
                     padding='valid',
                     activation={{choice(['relu', 'sigmoid'])}},
                     strides=1))

    model.add(GlobalMaxPooling1D(input_shape=train_data.shape))

    #model.add(Dense({{choice([25, 50, 100, 250])}}, input_shape=(3,)))
    #model.add(Dropout({{uniform(0, 1)}}))
    #model.add(Activation({{choice(['relu', 'sigmoid'])}}))

    #model.add(Dense({{choice([25, 50, 100, 250])}}, input_shape=(3,)))
    #model.add(Activation({{choice(['relu', 'sigmoid'])}}))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    model.fit(train_data, train_labels,
              batch_size={{choice([16, 32, 64, 128])}},
              epochs={{choice([1, 3, 5, 10])}},
              validation_data=(test_data, test_labels))

    score, acc = model.evaluate(test_data, test_labels, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


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