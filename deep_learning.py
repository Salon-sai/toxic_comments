# -*- coding: utf-8 -*-

import argparse
import os
import sys
import datetime
import gc

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold

from keras import callbacks
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from evaluate import RocAucCheckpoint
from model import deeplearning_models

from data_process import utils
# from subprocess import check_output
# print(check_output(["ls", "./input"]).decode("utf8"))

def main(args):
    gc.enable()
    subdir = datetime.datetime.strftime(datetime.datetime.now(), '[<(%Y%m%d-%H%M%S)>]')
    model_path = os.path.join(os.path.expanduser(args.models_path), "_".join([args.sub_models_path_base_name, subdir]))
    pretrained_word2vec = os.path.join(os.path.expanduser(args.pretrained_word2vec))

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    train_df = pd.read_csv(os.path.expanduser(args.train_data_path))
    test_df = pd.read_csv(os.path.expanduser(args.test_data_path))

    list_classes = train_df.columns[2:]
    indices = np.arange(train_df.shape[0])
    y = train_df[list_classes].values

    X_train, X_test, tokenizer = sentences2tokenized(train_df['comment_text'].fillna("CVxTz").values,
                                                     test_df['comment_text'].fillna("CVxTz").values,
                                                     args.max_features,
                                                     args.max_length)

    embedding_matrix = generate_embedding_matrix(pretrained_word2vec, args.embeddings_size, tokenizer, args.max_features)


    early = EarlyStopping(monitor="val_loss", mode="min", patience=2)

    if args.mode == 'STACK_PREPARE':
        k_fold = KFold(n_splits=args.n_split)
        train_id = pd.DataFrame({"id": train_df["id"]})
        preds_valid_train = np.zeros((train_df.shape[0], len(list_classes)), dtype=np.float32)
        preds_test = np.zeros((args.n_split, test_df.shape[0], len(list_classes)), dtype=np.float32)
        for k_fold_index, (train_indices, valid_indices) in enumerate(k_fold.split(indices)):
            model = getattr(deeplearning_models, args.model_def)(max_length=args.max_length,
                                                                 max_features=min(args.max_features, len(tokenizer.word_index)),
                                                                 embeddings_size=args.embeddings_size,
                                                                 embeddings_weights=embedding_matrix,
                                                                 num_units=args.num_units)
            model_fold_path = os.path.join(model_path, "fold_%d" % k_fold_index)
            if not os.path.exists(model_fold_path):
                os.mkdir(model_fold_path)
            train_x = X_train[train_indices, :]
            train_y = y[train_indices, :]
            valid_x = X_train[valid_indices, :]
            valid_y = y[valid_indices, :]

            print("The %d-th fold" % k_fold_index)
            ra_val = RocAucCheckpoint(file_path=os.path.join(model_fold_path, args.save_model_file),
                                      validation_data=(valid_x, valid_y), interval=1)
            checkpoint = ModelCheckpoint(os.path.join(model_fold_path, args.save_model_file), monitor="val_acc",
                                         verbose=1, save_best_only=True, mode='max')
            model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.epoch_size,
                      validation_data=(valid_x, valid_y), callbacks=[early, ra_val, checkpoint])
            # load the best model in training time
            model.load_weights(os.path.join(model_fold_path, args.save_model_file))
            preds_valid_train[valid_indices, :] = model.predict(valid_x, batch_size=args.batch_size, verbose=1)
            preds_test[k_fold_index] = model.predict(X_test, batch_size=args.batch_size, verbose=1)
        preds_test_mean = preds_test.mean(axis=0)
        sample_submission = pd.read_csv(args.sample_submission)
        sample_submission[list_classes] = preds_test_mean
        sample_submission.to_csv('./results/%s_%s_%d_fold_avg.csv' %
                                 (args.model_def, os.path.basename(pretrained_word2vec), k_fold.n_splits), index=False)
        train_predicts = pd.concat([train_id, pd.DataFrame(preds_valid_train, columns=list_classes)], axis=1)
        train_predicts.to_csv("./stacking_prepare/%s_%s_%s_train_predicts.csv" %
                              (args.model_def, args.rnn_type, os.path.basename(pretrained_word2vec)), index=False)

    if args.mode == 'TRAIN':
        [x_train, x_val, y_train, y_val] = train_test_split(X_train, y, train_size=0.9)

        model_file_path = os.path.join(model_path, args.save_model_file)
        model = getattr(deeplearning_models, args.model_def)(rnn_type=args.rnn_type,
                                                             max_length=args.max_length,
                                                             max_features=min(args.max_features, len(tokenizer.word_index)),
                                                             embeddings_size= args.embeddings_size,
                                                             embeddings_weights=embedding_matrix,
                                                             num_units=args.num_units)
        lr = callbacks.LearningRateScheduler(learning_rate_schedule)
        checkpoint = ModelCheckpoint(model_file_path, monitor="val_acc", verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
        ra_val = RocAucCheckpoint(file_path=model_file_path, validation_data=(x_val, y_val), interval=1)

        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epoch_size, validation_data=(x_val, y_val),
                  # callbacks=[lr, early, ra_val])
                  callbacks=[early, ra_val, checkpoint])

        model.load_weights(model_file_path)
        predict_save(model, X_test, args.sample_submission, args.result_file, list_classes, args.batch_size)
    elif args.mode == 'ALL_TRAIN':
        model = getattr(deeplearning_models, args.model_def)(max_length=args.max_length,
                                                             max_features=min(args.max_features, len(tokenizer.word_index)),
                                                             embeddings_size=args.embeddings_size,
                                                             embeddings_weights=embedding_matrix,
                                                             num_units=args.num_units)
        model = training(X_train, y, model, model_path, args.save_model_file, args.batch_size, args.epoch_size)
        predict_save(model, X_test, args.sample_submission, args.result_file, list_classes, args.batch_size)
    elif args.mode == 'TEST':
        model = getattr(deeplearning_models, args.model_def)(rnn_type=args.rnn_type,
                                                             max_length=args.max_length,
                                                             max_features=min(args.max_features, len(tokenizer.word_index)),
                                                             embeddings_size= args.embeddings_size,
                                                             embeddings_weights=embedding_matrix,
                                                             num_units=args.num_units)
        model.load_weights(args.load_freeze_model_file)
        predict_save(model, X_test, args.sample_submission, args.result_file, list_classes, args.batch_size)

def training(X, y, model, model_path, save_model_file, batch_size, epoch_size):
    model_file_path = os.path.join(model_path, save_model_file)
    checkpoint = ModelCheckpoint(model_file_path, monitor='acc', mode="max")
    model.fit(X, y, batch_size=batch_size, epochs=epoch_size, callbacks=[checkpoint])
    return model

def predict_save(model, X_test, sample_submission_csv, result_file, label_names, batch_size):
    y_predict = model.predict(X_test, batch_size=batch_size, verbose=1)
    sample_submission = pd.read_csv(sample_submission_csv)
    sample_submission[label_names] = y_predict
    sample_submission.to_csv(result_file, index=False)

# def test_with_freezemodel(X_test, rnn_type, model_def, max_length, max_features, tokenizer, embeddings_size,
#                           embedding_matrix, num_units, load_freeze_model_file, result_file, sample_submission,
#                           list_classes, batch_size):
#     model = getattr(deeplearning_models, model_def)(rnn_type=rnn_type,
#                                                          max_length=max_length,
#                                                          max_features=min(max_features, len(tokenizer.word_index)),
#                                                          embeddings_size=embeddings_size,
#                                                          embeddings_weights=embedding_matrix,
#                                                          num_units=num_units)
#     if os.path.isfile(load_freeze_model_file):
#         # just a single model
#         model.load_weights(load_freeze_model_file)
#         predict_save(model, X_test, sample_submission, result_file, list_classes, batch_size)
#     elif os.path.isdir(load_freeze_model_file):
#         pass

def generate_embedding_matrix(pretrained_word2vec, embeddings_size, tokenizer, max_features):
    embeddings_dict = dict()
    with open(pretrained_word2vec, encoding="utf8") as f:
        for index, line in enumerate(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            if len(values[1:]) != embeddings_size:
                print("The %d line is not enough dimension, the word : %s" % ((index + 1), word))
            else:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs

    all_embeddings = np.stack(embeddings_dict.values())
    print("The shape of pre-trained embeddings", all_embeddings.shape)
    embeddings_mean, embedding_std = all_embeddings.mean(), all_embeddings.std()
    del all_embeddings
    gc.collect()
    word_index = tokenizer.word_index
    print("the unique words in train comments is %d" % len(word_index))
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(embeddings_mean, embedding_std, (nb_words, embeddings_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def sentences2tokenized(train_comments, test_comments, max_words, max_length):
    new_train_comments = []
    new_test_comments = []
    print("pre-process the comments")
    for comment in train_comments:
        new_train_comments.append(utils.expression2word(comment))
    for comment in test_comments:
        new_test_comments.append(utils.expression2word(comment))

    tokenizer = text.Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(list(new_train_comments) + list(new_test_comments))

    list_tokenized_train = tokenizer.texts_to_sequences(new_train_comments)
    list_tokenized_test = tokenizer.texts_to_sequences(new_test_comments)

    # align all sequences
    x_train = sequence.pad_sequences(list_tokenized_train, max_length)
    x_test = sequence.pad_sequences(list_tokenized_test, max_length)
    return x_train, x_test, tokenizer

def learning_rate_schedule(ind):
    a = [0.002, 0.001, 0.00005, 0.00001, 0.00001]
    return a[ind]

def get_coefs(word,*arr):
    try:
        embedding = np.asarray(arr, dtype=np.float32)
        return word, embedding
    except ValueError:
        print(word, arr[:-300],len(arr))
        return None, None

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of test", default="./input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of train", default="./input/train.csv")
    parser.add_argument("--mode", type=str, choices=['TRAIN', 'TEST', 'STACK_PREPARE', "ALL_TRAIN"],
                        help="The mode of train or test", default="TRAIN")
    parser.add_argument("--n_split", type=int, help="The number of split data", default=5)
    parser.add_argument("--embeddings_size", type=int, help="The size of word embeddings", default=300)
    parser.add_argument("--num_units", type=int, help="The number of RNN units", default=128)
    parser.add_argument("--max_features", type=int, help="The max number of features", default=300000)
    parser.add_argument("--max_length", type=int, help="The max length of comment", default=200)
    parser.add_argument("--batch_size", type=int, help="The size of batch", default=128)
    parser.add_argument("--epoch_size", type=int, help="The size of epoch", default=3)
    parser.add_argument("--models_path", type=str, help="The path of saving model", default="./trained_models")
    parser.add_argument("--sub_models_path_base_name", type=str, help="The sub-path of saving model",
                        default="lstm-GolVe-300")
    parser.add_argument("--save_model_file", type=str, help="The file name of freeze model",
                        default="weights_base.best.hdf")
    parser.add_argument("--model_def", type=str, help="The models hoped to invoke in rnn module",
                        default="RNN_Conv1d")
    parser.add_argument("--rnn_type", type=str, choices=["lstm", "gru"], help="The type of RNN model", default="gru")
    parser.add_argument("--load_freeze_model_file", type=str, help="The file of freeze model",
                        default=None)
    parser.add_argument("--pretrained_word2vec", type=str, help="The path GloVe words vector",
                        default="./input/glove.840B.300d.txt")
    parser.add_argument("--sample_submission", type=str, help="The input sample submission file",
                        default="./input/sample_submission.csv")
    parser.add_argument("--result_file", type=str, help="The result of submission file",
                        default="./results/glove_lstm_baseline.csv")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

