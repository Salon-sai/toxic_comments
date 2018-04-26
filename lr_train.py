# -*- coding: utf-8 -*-

import sys
import argparse
import os
import gc

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy import sparse
from model.NB_SVM import NbSvmClassifier

def main(args):
    gc.enable()
    assert len(args.char_ngram_range) == 2
    char_ngram_range = tuple(args.char_ngram_range)

    test_submission_file = os.path.join("./results", args.result_submission_file)
    valid_stack_file = os.path.join("./stacking_prepare", args.stacking_file)
    train_df = pd.read_csv(os.path.join(os.path.expanduser(args.train_data_path)))
    test_df = pd.read_csv(os.path.join(os.path.expanduser(args.test_data_path)))

    column_names = train_df.columns[2:].values

    print('Convert the TF-IDF features of words....')
    train_words, test_words = load_words_features(train_df['comment_text'].tolist(), test_df['comment_text'].tolist(),
                                                  args.max_vocab)
    print('Convert the TF-IDF features of chars....')
    train_chars, test_chars = load_chars_features(train_df['comment_text'].tolist(), test_df['comment_text'].tolist(),
                                                  args.max_n_char, char_ngram_range)

    train_feats = sparse.hstack([train_words, train_chars], format='csr')
    test_feats = sparse.hstack([test_words, test_chars], format='csr')

    valid_preds, test_avg_preds = valid_stack_prepare(train_feats, train_df[column_names].values, test_feats, 
                                                      column_names, args.n_splits)
    train_stack_df = pd.DataFrame({"id": train_df["id"]})
    train_stack_df = pd.concat([train_stack_df, pd.DataFrame(valid_preds, columns=column_names)], axis=1)
    train_stack_df.to_csv(valid_stack_file, index=False)

    preds = full_train_predict(train_feats, train_df[column_names].values, test_feats, column_names)
    submission_id = pd.DataFrame({"id": test_df["id"]})
    submission = pd.concat([submission_id, pd.DataFrame(preds, columns=column_names)], axis=1)
    submission.to_csv(test_submission_file, index=False)

    submission = pd.concat([submission_id, pd.DataFrame(test_avg_preds, columns=column_names)], axis=1)
    submission.to_csv("./results/lr_tf_idf_5_fold_avg.csv", index=False)

def valid_stack_prepare(train_features, train_targets, test_features, column_names, n_splits=5):
    k_fold = KFold(n_splits=n_splits)
    # predict the valid data splitting from training data
    valid_preds = np.zeros((train_features.shape[0], len(column_names)), dtype=np.float32)
    # predict the test data use splitting train data
    test_fold_preds = np.zeros((n_splits, test_features.shape[0], len(column_names)), dtype=np.float32)
    print("validating and stacking")
    indices = np.arange(train_features.shape[0])
    for i, label in enumerate(column_names):
        target = train_targets[:, i]
        auc_scores = np.zeros((n_splits))
        print("Start fold validate %s" % label)
        for fold_index, (train_index, valid_index) in enumerate(k_fold.split(indices)):
            x_train = train_features[train_index, :]
            y_train = target[train_index]
            x_valid = train_features[valid_index, :]
            y_valid = target[valid_index]
            model = LogisticRegression(solver='sag', n_jobs=-1)
            # model = NbSvmClassifier(C=4.0, dual=True)
            model.fit(x_train, y_train)
            valid_preds[valid_index, i] = model.predict_proba(x_valid)[:, 1]
            test_fold_preds[fold_index, :, i] = model.predict_proba(test_features)[:, 1]
            score = roc_auc_score(y_valid, valid_preds[valid_index, i])
            print("\t Fold %d validating, score: %.8f" % (fold_index, score))
            auc_scores[fold_index] = score
        full_score_i = roc_auc_score(target, valid_preds[:, i])
        print("\t Avg AUC score %.8f, Full AUC score: %.8f" % (auc_scores.mean(), full_score_i))
    print("\t Final Full AUC %.8f" % roc_auc_score(train_targets, valid_preds))
    return valid_preds, test_fold_preds.mean(axis=0)

def full_train_predict(train_features, train_target, test_features, column_names):
    print("Full training...")
    predicts = np.zeros((test_features.shape[0], len(column_names)), dtype=np.float32)
    for i, label in enumerate(column_names):
        print("\t Full Training the %s ...." % label)
        model = LogisticRegression(solver='sag', n_jobs=-1)
        # model = NbSvmClassifier(C=4.0, dual=True)
        model.fit(train_features, train_target[:, i])
        print("\t Predicting on test ")
        predicts[:, i] = model.predict_proba(test_features)[:, 1]
    return predicts

def load_words_features(train_comments, test_comments, max_vocab):
    all_comments = train_comments + test_comments
    vect_words = TfidfVectorizer(ngram_range=(1, 1),
                                 strip_accents='unicode',
                                 max_features=max_vocab,
                                 token_pattern=r'\w{1,}',
                                 analyzer='word',
                                 stop_words='english',
                                 sublinear_tf=True)
    vector = vect_words.fit(all_comments)
    train_features = vector.transform(train_comments)
    test_features = vector.transform(test_comments)
    return train_features, test_features

def load_chars_features(train_comments, test_comments, max_n_char, char_ngram_range):
    all_comments = train_comments + test_comments
    vect_chars = TfidfVectorizer(ngram_range=char_ngram_range,
                                 max_features=max_n_char,
                                 analyzer='char',
                                 stop_words='english',
                                 strip_accents='unicode',
                                 sublinear_tf=True)
    vector = vect_chars.fit(all_comments)
    train_features = vector.transform(train_comments)
    test_features = vector.transform(test_comments)
    return train_features, test_features

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of test", default="./input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of train", default="./input/train.csv")
    parser.add_argument("--max_vocab", type=int, help="The max vocab of tf-idf features", default=10000)
    parser.add_argument("--max_n_char", type=int, help="The max n-chars of tf-idf features", default=50000)
    parser.add_argument("--train_with_convai_path", type=str, help="The train csv of toxic comments with predict of "
                                                                   "features from conversationai dataset",
                        default="./input/train_with_convai.csv")
    parser.add_argument("--test_with_convai_path", type=str, help="The test csv of toxic comments with predict of "
                                                                  "features from conversationai dataset",
                        default="./input/test_with_convai.csv")
    parser.add_argument("--result_submission_file", type=str, help="The submission result of test predict",
                        default="lr_tf_idf.csv")
    parser.add_argument("--stacking_file", type=str, help="The stacking file of train", default="lr_tf_idf.csv")
    parser.add_argument("--n_splits", type=int, help="The number of splits", default=5)
    parser.add_argument("--char_ngram_range", type=int, nargs="+", help="ngram range of chars", default=(2, 6))
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


