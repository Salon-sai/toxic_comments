# -*- coding: utf-8 -*-

import os
import sys
import argparse

import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPool1D
from sklearn.model_selection import train_test_split


def main(args):
    list_of_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    train_df = pd.read_csv("./input/train.csv")

    print(np.where(pd.isnull(train_df)))

    x = train_df['comment_text'].values

    print("properties of x")
    print("type : {}, dimensions : {}, shape : {}, total no. of elements : {}, data type of each element: {}, size of "
          "each element {} bytes".format(type(x), x.ndim, x.shape, x.size, x.dtype, x.itemsize))

    y = train_df[list_of_classes].values

    x_tokenizer = text.Tokenizer(num_words=args.max_features)
    x_tokenizer.fit_on_texts(list(x))
    x_tokenized = x_tokenizer.texts_to_sequences(x)

    x_train_val = sequence.pad_sequences(x_tokenized, maxlen=args.max_text_length)

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.1, random_state=1)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(args.max_features, args.embedding_dims, input_length=args.max_text_length))
    model.add(Dropout(0.2))
    model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())

    model.add(Dense(250))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(6))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epoch_size,
              validation_data=(x_val, y_val))

    test_df = pd.read_csv('./input/test.csv')
    print(np.where(pd.isnull(test_df)))
    x_test = test_df['comment_text'].fillna('comment_missing').values

    x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
    x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=args.max_text_length)
    y_testing = model.predict(x_testing, verbose=1)

    sample_submission = pd.read_csv("./input/sample_submission.csv")
    sample_submission[list_of_classes] = y_testing
    sample_submission.to_csv(args.result_file, index=False)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, help="The max number of features", default=2000)
    parser.add_argument("--max_text_length", type=int, help="The max length of comment text", default=400)
    parser.add_argument("--embedding_dims", type=int, help="The dimensions of embedding", default=50)
    parser.add_argument("--num_filter", type=int, help="The number of conv1D filters", default=250)
    parser.add_argument("--batch_size", type=int, help="The size of batch", default=32)
    parser.add_argument("--epoch_size", type=int, help="The size of batch", default=2)
    parser.add_argument("--result_file", type=str, help="The result of submission file",
                        default="./result/cnn_baseline.csv")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# list_of_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# max_features = 20000
# max_text_length = 400
# embedding_dims = 50
# filters = 250
# kernel_size = 3
# hidden_dims = 250
# batch_size = 32
# epochs = 30




