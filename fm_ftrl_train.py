# -*- coding: utf-8 -*-

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import regex as re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from wordbatch.models import FM_FTRL

from data_process.utils import char_analyzer
from feature_engineer import indirect_features


@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(args):
    gc.enable()
    with timer("Reading input files"):
        train_data = pd.read_csv(args.train_data_path)
        test_data = pd.read_csv(args.test_data_path)

    label_names = train_data.columns[2:]

    with timer("Performing basic NLP"):
        indirect_features.get_indicators_and_clean_comments(train_data)
        indirect_features.get_indicators_and_clean_comments(test_data)

    with timer("Creating numerical features"):
        train_num_features, test_num_features = extract_num_features(train_data, test_data, label_names)

    train_text = train_data['clean_comment']
    test_text = test_data['clean_comment']
    all_text = pd.concat([train_text, test_text])

    with timer("Tfidf on word"):
        train_tf_idf, test_tf_idf = tf_idf_features(train_comments=train_text, test_comments=test_text)

    del train_text
    del test_text
    del all_text
    gc.collect()

    with timer("Stacking matrices"):
        all_train_features = hstack([train_tf_idf, train_num_features]).tocsr()
        del train_tf_idf
        del train_num_features
        gc.collect()

        all_test_features = hstack([test_tf_idf, test_num_features]).tocsr()
        del test_tf_idf
        del test_num_features
        gc.collect()

    print('Shapes jues to be sure: ', all_train_features.shape, all_test_features.shape)

    with timer("Scoring LogisticRegression"):
        valid_score(train_data, all_train_features, label_names, args.stacking_save_csv, args.n_splits)

    with timer("Full Training and predict test result"):
        predict_and_save(all_train_features, train_data[label_names].values, test_data, all_test_features, label_names,
                         args.submission_file)


def valid_score(train_data, all_train_features, label_names, stacking_save_csv, n_splits):
    folds = KFold(n_splits=n_splits)
    indices = np.arange(train_data.shape[0])
    stacking_submission = pd.DataFrame.from_dict({"id": train_data["id"]})
    valid_predicts = np.zeros((train_data.shape[0], len(label_names)))
    class_avg_score = np.zeros(len(label_names))
    class_weights = {'toxic': 1.0, 'severe_toxic': 0.2, 'obscene': 1.0, 'threat': 0.1, 'insult': 0.8,
                     'identity_hate': 0.2}
    for label_index, label_name in enumerate(label_names):
        label_target = train_data[label_name].values
        class_scores = np.zeros(folds.n_splits)
        print("Valid the %s" % label_name)
        train_weight = np.array([1.0 if x == 1 else class_weights[label_name] for x in label_target])
        for fold_index, (train_index, valid_index) in enumerate(folds.split(indices)):
            clf = FM_FTRL(
                alpha=0.02, beta=0.01, L1=0.00001, L2=30.0,
                D=all_train_features.shape[1], alpha_fm=0.1,
                L2_fm=0.5, init_fm=0.01, weight_fm=50.0,
                D_fm=200, e_noise=0.0, iters=3,
                inv_link="identity", e_clip=1.0, threads=4, use_avx=1, verbose=1
            )
            clf.fit(all_train_features[train_index], label_target[train_index], train_weight[train_index], reset=False)
            valid_predicts[valid_index, label_index] = sigmoid(clf.predict(all_train_features[valid_index]))
            score = roc_auc_score(label_target[valid_index], valid_predicts[valid_index, label_index])
            class_scores[fold_index] = score
            print("\t The Fold %d score %.8f" % (fold_index, score))
        # save the validate predict of training data
        stacking_submission[label_name] = valid_predicts[:, label_index]
        # compute the score (AUC) of current label
        full_score_class = roc_auc_score(label_target, valid_predicts[:, label_index])
        avg_score_class = class_scores.mean()
        class_avg_score[label_index] = avg_score_class
        print("\tThe %s class, Avg AUC %.8f, Full AUC %.8f" % (label_name, avg_score_class, full_score_class))
    # save the stacking file
    stacking_submission.to_csv(os.path.join("./stacking_prepare", stacking_save_csv), index=False)
    # compute the score of all train data
    all_full_score = roc_auc_score(train_data[label_names].values, valid_predicts)
    avg_score = class_avg_score.mean()
    print("All the Full AUC: %.8f, Avg AUC in all data: %.8f" % (all_full_score, avg_score))

def predict_and_save(all_train_features, train_labels, test_data, all_test_features, label_names, submission_file):
    submission = pd.DataFrame.from_dict({'id': test_data['id']})
    test_predict = np.zeros((test_data.shape[0], len(label_names)))
    class_weights = {'toxic': 1.0, 'severe_toxic': 0.2, 'obscene': 1.0, 'threat': 0.1, 'insult': 0.8,
                     'identity_hate': 0.2}
    for index, label_name in enumerate(label_names):
        label_target = train_labels[:, index]
        train_weight = np.array([1.0 if x == 1 else class_weights[label_name] for x in label_target])
        clf = FM_FTRL(
            alpha=0.02, beta=0.01, L1=0.00001, L2=30.0,
            D=all_train_features.shape[1], alpha_fm=0.1,
            L2_fm=0.5, init_fm=0.01, weight_fm=50.0,
            D_fm=200, e_noise=0.0, iters=3,
            inv_link="identity", e_clip=1.0, threads=4, use_avx=1, verbose=1
        )
        clf.fit(all_train_features, label_target, train_weight, reset=False)
        test_predict[:, index] = sigmoid(clf.predict(all_test_features))
        submission[label_name] = test_predict[:, index]
    submission.to_csv(os.path.join("./results", submission_file), index=False)


def tf_idf_features(train_comments, test_comments):
    all_comments = pd.concat([train_comments, test_comments])
    with timer("TF-IDF on words"):
        word_vetorizer = TfidfVectorizer(sublinear_tf=True,
                                         strip_accents='unicode',
                                         tokenizer=lambda x: re.findall(r'[^\p{P}\W]+', x),
                                         analyzer='word',
                                         token_pattern=None,
                                         stop_words='english',
                                         ngram_range=(1, 2),
                                         max_features=300000)
        with timer("Fit TF-IDF with words"):
            word_vetorizer.fit(all_comments)
        with timer("Transform the words in train set"):
            train_words_tf_idf = word_vetorizer.transform(train_comments)
        with timer("Transform the words in test set"):
            test_words_tf_idf = word_vetorizer.transform(test_comments)

    with timer("TF-IDF on chars"):
        char_vetorizer = TfidfVectorizer(sublinear_tf=True,
                                         strip_accents='unicode',
                                         analyzer="word",
                                         tokenizer=char_analyzer,
                                         ngram_range=(1, 3),
                                         max_features=60000)
        with timer("Fit TF-IDF with chars"):
            char_vetorizer.fit(all_comments)
        with timer("Transform the chars in train set"):
            train_chars_tf_idf = char_vetorizer.transform(train_comments)
        with timer("Transform the chars in test set"):
            test_chars_tf_idf = char_vetorizer.transform(test_comments)
    return hstack([train_words_tf_idf, train_chars_tf_idf]).tocsr(), \
           hstack([test_words_tf_idf, test_chars_tf_idf]).tocsr()

def extract_num_features(train_data, test_data, label_names):
    num_features = [f_ for f_ in train_data.columns
                    if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                  'has_ip_address'] + label_names.tolist()]
    for f in num_features:
        all_cut = pd.cut(pd.concat([train_data[f], test_data[f]], axis=0), bins=20, labels=False, retbins=False)
        train_data[f] = all_cut.values[:train_data.shape[0]]
        test_data[f] = all_cut.values[train_data.shape[0]:]
    return get_numerical_features(train_data[num_features], test_data[num_features])

def get_numerical_features(train_features, test_features):
    """
    As @bangda suggested FM_FTRL either needs to scaled output or dummies
    So here we go for dummies
    """
    ohe = OneHotEncoder()
    full_csr = ohe.fit_transform(np.vstack((train_features.values, test_features.values)))
    csr_train = full_csr[:train_features.shape[0]]
    csr_test = full_csr[train_features.shape[0]:]
    del full_csr
    gc.collect()
    # Now remove features that don't have enough samples either in train or test
    return clean_csr(csr_train, csr_test, 3)

def clean_csr(csr_train, csr_test, min_df):
    # remove some features that don't have enough samples
    trn_min = np.where(csr_train.getnnz(axis=0) >= min_df)[0]
    test_min = {x for x in np.where(csr_test.getnnz(axis=0) >= min_df)[0]}
    mask = [x for x in trn_min if x in test_min]
    return csr_train[:, mask], csr_test[:, mask]

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of original test", default="./input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of original train", default="./input/train.csv")
    parser.add_argument("--n_splits", type=int, help="The number of split data", default=5)
    parser.add_argument("--stacking_save_csv", type=str, help="The file name of stacking saving",
                        default="fm_ftrl_clean.csv")
    parser.add_argument("--submission_file", type=str, help="The file name of submssion by predict test data",
                        default="fm_ftrl_submission.csv")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
