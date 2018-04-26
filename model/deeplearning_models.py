# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Dense, Embedding, Input, SpatialDropout1D, concatenate
from keras.layers import LSTM, Bidirectional, Dropout, GRU, Conv1D, Conv2D
from keras.layers import GlobalMaxPool1D, GlobalAvgPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam

# def get_base_model(max_length, max_features, embeddings_size=128):
def get_base_model(**kwargs):
    if "max_length" not in kwargs.keys():
        raise ValueError('we need the max_length value')
    if "max_features" not in kwargs.keys():
        raise ValueError('we need the max_features value')
    if "embeddings_size" not in kwargs.keys():
        kwargs['embeddings_size'] = 128

    input = Input(shape=(kwargs.get("max_length"), ))
    x = Embedding(kwargs.get("max_features"), kwargs.get("embeddings_size"))(input)
    x = Bidirectional(LSTM(kwargs.get("num_units", 50), return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model

# def get_model_with_dropout(max_length, max_features, embeddings_weights, embeddings_size=128):
def get_model_with_dropout(**kwargs):
    if "max_length" not in kwargs.keys():
        raise ValueError('we need the max_length value')
    if "max_features" not in kwargs.keys():
        raise ValueError('we need the max_features value')
    if "embeddings_weights" not in kwargs.keys():
        raise ValueError('we need the embeddings_weights value')
    if "embeddings_size" not in kwargs.keys():
        kwargs['embeddings_size'] = 128

    input = Input(shape=(kwargs.get("max_length"),))
    x = Embedding(kwargs.get("max_features"), kwargs.get("embeddings_size"), weights=[kwargs.get("embeddings_weights")])(input)
    x = Bidirectional(LSTM(kwargs.get("num_units", 50), return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model

# def get_model_with_dropout_and_bn(max_length, max_features, embeddings_weights, embeddings_size=128):
def RNN_baseline(rnn_type="lstm", **kwargs):
    if "max_length" not in kwargs.keys():
        raise ValueError('we need the max_length value')
    if "max_features" not in kwargs.keys():
        raise ValueError('we need the max_features value')
    if "embeddings_weights" not in kwargs.keys():
        raise ValueError('we need the embeddings_weights value')
    if "embeddings_size" not in kwargs.keys():
        kwargs['embeddings_size'] = 128

    if rnn_type == 'gru':
        rnn_model = GRU(kwargs.get('num_units', 128), return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
    elif rnn_type == 'lstm':
        rnn_model = LSTM(kwargs.get("num_units", 128), return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
    else:
        raise ValueError("there are no %s model" % rnn_type)

    input = Input(shape=(kwargs.get('max_length'),))
    x = Embedding(kwargs.get('max_features'), kwargs.get('embeddings_size'), weights=[kwargs.get('embeddings_weights')])(input)
    x = Bidirectional(rnn_model)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model

def RNN_multi_pool(rnn_type="gru", **kwargs):
    if "max_length" not in kwargs.keys():
        raise ValueError('we need the max_length value')
    if "max_features" not in kwargs.keys():
        raise ValueError('we need the max_features value')
    if "embeddings_weights" not in kwargs.keys():
        raise ValueError('we need the embeddings_weights value')
    if "embeddings_size" not in kwargs.keys():
        kwargs['embeddings_size'] = 128

    if rnn_type == 'gru':
        rnn_model = GRU(kwargs.get('num_units', 128), return_sequences=True)
    elif rnn_type == 'lstm':
        rnn_model = LSTM(kwargs.get('num_units', 128), return_sequences=True)
    else:
        raise ValueError("there are no %s model" % rnn_type)

    input = Input(shape=(kwargs.get('max_length'), ))
    x = Embedding(kwargs.get('max_features'), kwargs.get('embeddings_size'), weights=[kwargs.get('embeddings_weights')],
                  trainable=False)(input)
    x = SpatialDropout1D(0.2)(x)
    # x = Bidirectional(GRU(kwargs.get('num_units', 80), return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Bidirectional(rnn_model)(x)
    avg_pool = GlobalMaxPool1D()(x)
    max_pool = GlobalAvgPool1D()(x)
    concate = concatenate([avg_pool, max_pool])
    output = Dense(6, activation="sigmoid")(concate)

    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model

def RNN_Conv1d(rnn_type="gru", **kwargs):
    if "max_length" not in kwargs.keys():
        raise ValueError('we need the max_length value')
    if "max_features" not in kwargs.keys():
        raise ValueError('we need the max_features value')
    if "embeddings_weights" not in kwargs.keys():
        raise ValueError('we need the embeddings_weights value')
    if "embeddings_size" not in kwargs.keys():
        kwargs['embeddings_size'] = 128

    if rnn_type == 'gru':
        rnn_model = GRU(kwargs.get('num_units', 128), return_sequences=True)
    elif rnn_type == 'lstm':
        rnn_model = LSTM(kwargs.get('num_units', 128), return_sequences=True)
    else:
        raise ValueError("there are no %s model" % rnn_type)

    input = Input(shape=(kwargs.get('max_length'), ))
    x = Embedding(kwargs.get('max_features'), kwargs.get('embeddings_size'), weights=[kwargs.get('embeddings_weights')],
                  trainable=False)(input)
    x = SpatialDropout1D(0.2)(x)
    # x = Bidirectional(GRU(kwargs.get('num_units', 80), return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Bidirectional(rnn_model)(x)
    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    concate = concatenate([avg_pool, max_pool])
    output = Dense(6, activation="sigmoid")(concate)

    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model

def text_cnn_baseline(**kwargs):
    if "max_length" not in kwargs.keys():
        raise ValueError('we need the max_length value')
    if "max_features" not in kwargs.keys():
        raise ValueError('we need the max_features value')
    if "embeddings_weights" not in kwargs.keys():
        raise ValueError('we need the embeddings_weights value')
    if "embeddings_size" not in kwargs.keys():
        kwargs['embeddings_size'] = 128
    if "kernel_sizes" not in kwargs.keys():
        kwargs["kernel_sizes"] = [2, 3, 4, 5]

    input = Input(shape=(kwargs.get("max_length"), ))
    x = Embedding(kwargs.get('max_features'), kwargs.get('embeddings_size'),
                  weights=[kwargs.get('embeddings_weights')], trainable=False)(input)
    x = SpatialDropout1D(0.2)(x)

    concat_array = []
    for kernel_size in kwargs.get("kernel_sizes"):
        conv = Conv1D(filters=64, kernel_size=kernel_size, strides=1, padding="valid", activation="relu")(x)
        avg_pool = GlobalAveragePooling1D()(conv)
        max_pool = GlobalMaxPooling1D()(conv)
        concat_array += [avg_pool, max_pool]
    concat = concatenate(concat_array)
    output = Dense(6, activation="sigmoid")(concat)

    model = Model(inputs=input, outputs=output)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    return model
