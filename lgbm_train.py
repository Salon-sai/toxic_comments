# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import gc

from collections import defaultdict

import pandas as pd
import numpy as np

import lightgbm as lgb

from contextlib import contextmanager

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from feature_engineer import indirect_features
from tf_idf_train import generate_tf_idf_features
from data_process.utils import char_analyzer

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
        # train_tf_idf, test_tf_idf = generate_tf_idf_features(all_text, train_data.shape[0], test_data.shape[0], None,
        #                                                      False, count_min_tf=0, count_max_tf=1,
        #                                                      need_char_features=False, char_ngram_range=(1, 3),
        #                                                      word_ngram_range=(1, 2), tokenizer="tokenize")
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

    params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1
    }

    with timer("Scoring Light GBM"):
        lgb_round_dict = defaultdict(int)
        train_lgbset = lgb.Dataset(all_train_features, free_raw_data=False)
        del all_train_features
        gc.collect()
        stack_preds = pd.DataFrame({"id": train_data["id"]})
        scores = []  # scores of different labels
        for label_name in label_names:
            print("Class %s scores :" % label_name)
            train_target = train_data[label_name]
            train_lgbset.set_label(train_target.values)
            preds, full_score = valid_train_lgm(params=params,
                                                train_lgbset=train_lgbset,
                                                train_data=train_data,
                                                label_name=label_name,
                                                lgb_round_dict=lgb_round_dict,
                                                n_split=args.n_split)
            scores.append(full_score)
            stack_preds[label_name] = preds
            stack_preds.to_csv(os.path.join("./stacking_prepare", args.stacking_save_csv), index=False,
                               float_format="%.8f")

        print("Total CV score is {}".format(np.mean(scores)))

    submission = pd.DataFrame({"id": test_data["id"]})
    with timer("Predicting probabilities"):
        for label_name in label_names:
            with timer("Predicting probabilities for %s" % label_name):
                train_target = train_data[label_name]
                train_lgbset.set_label(train_target.values)

                model = lgb.train(params=params,
                                  train_set=train_lgbset,
                                  num_boost_round=int(lgb_round_dict[label_name] / args.n_split))
                submission[label_name] = model.predict(all_test_features, num_iteration=model.best_iteration)
    submission.to_csv(args.submission_save_csv, index=False, float_format="%.8f")

def extract_num_features(train_data, test_data, label_names):
    num_features = [f_ for f_ in train_data.columns
                    if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                  'has_ip_address'] + label_names.tolist()]
    skl = MinMaxScaler()
    train_num_features = csr_matrix(skl.fit_transform(train_data[num_features]))
    test_num_features = csr_matrix(skl.fit_transform(test_data[num_features]))
    return train_num_features, test_num_features

def valid_train_lgm(params, train_lgbset, train_data, label_name, lgb_round_dict, n_split=5):
    folds = KFold(n_splits=n_split)
    train_target = train_data[label_name]
    train_lgbset.set_label(train_target.values)

    lgb_rounds = 1000
    class_pred = np.zeros(train_data.shape[0])

    for n_fold, (train_indices, valid_indices) in enumerate(folds.split(train_data, train_target)):
        watchlist = [train_lgbset.subset(train_indices), train_lgbset.subset(valid_indices)]

        model = lgb.train(params=params,
                          train_set=watchlist[0],
                          num_boost_round=lgb_rounds,
                          valid_sets=watchlist,
                          early_stopping_rounds=50,
                          verbose_eval=0)
        class_pred[valid_indices] = model.predict(train_lgbset.data[valid_indices], num_iteration=model.best_iteration)
        score = roc_auc_score(train_target.values[valid_indices], class_pred[valid_indices])
        # Compute mean rounds over folds for each class
        # So that it can be re-used for test predictions
        lgb_round_dict[label_name] += model.best_iteration
        print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, score, model.best_iteration))
    full_score = roc_auc_score(train_target, class_pred)
    print("full score : %.6f" % full_score)
    return class_pred, full_score

def tf_idf_features(train_comments, test_comments):
    all_comments = pd.concat([train_comments, test_comments])
    with timer("TF-IDF on words"):
        word_vetorizer = TfidfVectorizer(sublinear_tf=True,
                                         strip_accents='unicode',
                                         analyzer='word',
                                         token_pattern=r'\w{1,}',
                                         stop_words='english',
                                         ngram_range=(1, 2),
                                         max_features=20000)
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
                                         ngram_range=(1, 1),
                                         max_features=50000)
        with timer("Fit TF-IDF with chars"):
            char_vetorizer.fit(all_comments)
        with timer("Transform the chars in train set"):
            train_chars_tf_idf = char_vetorizer.transform(train_comments)
        with timer("Transform the chars in test set"):
            test_chars_tf_idf = char_vetorizer.transform(test_comments)
    return hstack([train_words_tf_idf, train_chars_tf_idf]).tocsr(), \
           hstack([test_words_tf_idf, test_chars_tf_idf]).tocsr()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of original test", default="./input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of original train", default="./input/train.csv")
    parser.add_argument("--n_split", type=int, help="The number of split data", default=5)
    parser.add_argument("--stacking_save_csv", type=str, help="The file name of stacking saving",
                        default="lvl0_lgbm_clean.csv")
    parser.add_argument("--submission_save_csv", type=str, help="The file name of submssion by predict test data", 
                        default="./results/lvl0_lgbm_clean.csv")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

