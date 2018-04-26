# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time

import pandas as pd
import numpy as np
import datetime

from scipy import sparse

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from feature_engineer.tf_idf_features_process import WordTfidfFeatures, CharTfidfFeatures, CountFeatures

import data_process.utils as tokenizer_utils

from model.NB_SVM import NbSvmClassifier
from data_process.extend_text import extend_text_by_markovify

def main(args):
    assert hasattr(tokenizer_utils, args.tokenizer)
    assert len(args.char_ngram_range) == 2
    assert len(args.word_ngram_range) == 2
    char_ngram_range = tuple(args.char_ngram_range)
    word_ngram_range = tuple(args.word_ngram_range)

    train = pd.read_csv(os.path.expanduser(args.train_data_path))
    test = pd.read_csv(os.path.expanduser(args.test_data_path))
    indices = np.arange(train.shape[0])
    label_names = train.columns[2: 8].values
    y = np.asarray(train[label_names].values)
    all_comments = train['comment_text'].tolist() + test['comment_text'].tolist()

    if args.need_convai_features:
        train_convai_features, test_convai_features = get_convai_features(args.train_convai_path, args.test_convai_path)
    else:
        train_convai_features = None
        test_convai_features = None

    if args.mode == 'VALID':
        train_features, _ = generate_tf_idf_features(all_comments, train.shape[0], test.shape[0],
                                                     args.train_indirect_features_path, args.need_count_features,
                                                     args.count_min_tf, args.count_max_tf, args.need_char_features,
                                                     char_ngram_range, word_ngram_range, args.tokenizer)

        train_features = sparse.hstack([train_features, train_convai_features], format="csr")

        k_fold = KFold(n_splits=args.n_split)
        classifiers = [NbSvmClassifier(C=4, dual=True), LogisticRegression(C=4, dual=True),
                       # RandomForestClassifier(n_estimators=20, n_jobs=-1),
                       # AdaBoostClassifier(n_estimators=100)
                       ]
        preds_scores = np.zeros((args.n_split, len(classifiers)))

        print("The train features shape is ", train_features.shape)

        for cls_index, classify in enumerate(classifiers):
            preds_valid_train = np.zeros((train.shape[0], len(label_names)), dtype=np.float32)
            for k_fold_index, (train_indices, valid_indices) in enumerate(k_fold.split(indices)):
                train_x = train_features[train_indices, :]
                train_y = y[train_indices, :]
                valid_x = train_features[valid_indices, :]
                valid_y = y[valid_indices, :]
                print("The %d-th fold" % k_fold_index)
            # for cls_index, classify in enumerate(classifiers):
                preds = np.zeros_like(valid_y, dtype=np.float32)
                start_time = time.time()
                for label_index, label_y in enumerate(train_y.transpose()):
                    print("Fit..%s with %s" % (label_names[label_index], type(classify).__name__))
                    classify.fit(train_x, label_y)
                    preds[:, label_index] = classify.predict_proba(valid_x)[:, 1]
                    preds_valid_train[valid_indices, label_index] = preds[:, label_index]
                preds_scores[k_fold_index, cls_index] = metrics.roc_auc_score(valid_y, preds)
                print("Train and valid with all label speed %1.5f" % (time.time() - start_time))
                print("--------------------------")
            submid = pd.DataFrame({'id': train['id']})
            submission = pd.concat([submid, pd.DataFrame(preds_valid_train, columns=label_names)], axis=1)
            submission.to_csv("./stacking_prepare/%s_train_preds_char_%d-%d.csv" %
                              (type(classify).__name__, char_ngram_range[0], char_ngram_range[1]), index=False)

        mean_scores = preds_scores.mean(axis=0)
        std_scores = preds_scores.std(axis=0)

        for i, (mean_score, std_score) in enumerate(zip(mean_scores, std_scores)):
            print("% s: the AUC score is %1.4f+-%1.4f " % (type(classifiers[i]).__name__ , mean_score, std_score))
    elif args.mode == 'PREDICT':
        subdir = datetime.datetime.strftime(datetime.datetime.now(), '[<(%Y%m%d-%H%M%S)>]')
        save_model_path = os.path.expanduser(os.path.join(args.save_model_path, subdir))
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        preds = np.zeros((test.shape[0], len(label_names)))
        for label_index, label_y in enumerate(y.transpose()):
            label_name = label_names[label_index]

            class_comments = train[train[label_name].eq(1)].comment_text
            extend_comments = extend_text_by_markovify(class_comments)
            extend_all_comments = all_comments + extend_comments

            train_features, test_features = generate_tf_idf_features(extend_all_comments, train.shape[0], test.shape[0],
                                                                     args.train_indirect_features_path,
                                                                     args.need_count_features, args.count_min_tf,
                                                                     args.count_max_tf, args.need_char_features,
                                                                     char_ngram_range, word_ngram_range, args.tokenizer)

            train_features = sparse.hstack([train_features, train_convai_features], format="csr")
            test_features = sparse.hstack([test_features, test_convai_features], format="csr")

            assert test_features.shape[0] == test.shape[0]
            if args.pretrained_model_path is None:
                classifier = NbSvmClassifier(C=4, dual=True)
                print("fit the ", label_name)
                classifier.fit(train_features, label_y)
                joblib.dump(classifier, save_model_path + "/" + "classifier" + "_" + label_name + ".pkl")
            else:
                print("loading the %s model" % label_name)
                classifier = joblib.load(os.path.expanduser(os.path.join(args.pretrained_model_path,
                                                                         "classifier" + "_" + label_name + ".pkl")))
            print("predict the %s" % label_name)
            preds[:, label_index] = classifier.predict_proba(test_features)[:, 1]
            train_features, test_features = None, None

        submid = pd.DataFrame({'id': test["id"]})
        submission = pd.concat([submid, pd.DataFrame(preds, columns=label_names)], axis=1)
        submission.to_csv(os.path.expanduser(args.result_submission_file), index=False)

def generate_tf_idf_features(comments, num_train, num_test, train_indirect_features_path, need_count_features, count_min_tf,
                             count_max_tf, need_char_features, char_ngram_range, word_ngram_range, tokenizer):
    # TODO: need to add indirect features for test data
    if train_indirect_features_path is not None:
        train_indirect_features = pd.read_csv(os.path.expanduser(train_indirect_features_path))
        indirect_feature_names = train_indirect_features.columns[1:].values
        train_indirect_features = train_indirect_features[indirect_feature_names].values
    else:
        train_indirect_features = None

    print("converting word tf-idf features.....")
    word_process = WordTfidfFeatures(comments, tokenizer=getattr(tokenizer_utils, tokenizer),
                                     ngram_range=word_ngram_range)
    word_features = word_process.tf_idf_features

    if need_char_features:
        print("converting char tf-idf features.....")
        char_process = CharTfidfFeatures(comments, ngram_range=char_ngram_range)
        char_features = char_process.tf_idf_features
    else:
        char_features = None

    if need_count_features:
        print("converting count features.....")
        count_process = CountFeatures(comments, min_df=count_min_tf, max_df=count_max_tf)
        count_features = count_process.count_features
    else:
        count_features = None

    train_features = sparse.hstack([
        word_features[:num_train],
        char_features[:num_train] if char_features is not None else None,
        count_features[:num_train] if count_features is not None else None], format="csr")
    test_features = sparse.hstack([
        word_features[num_train:num_train + num_test],
        char_features[num_train:num_train + num_test] if char_features is not None else None,
        count_features[num_train:num_train + num_test] if count_features is not None else None], format="csr")

    return train_features, test_features


def get_convai_features(train_convai_csv, test_convai_csv):
    train_data = pd.read_csv(train_convai_csv)
    test_data = pd.read_csv(test_convai_csv)
    feature_names = ['toxic_level', 'attack', 'aggression']
    return train_data[feature_names], test_data[feature_names]


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of original test", default="./input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of original train", default="./input/train.csv")
    parser.add_argument("--test_convai_path", type=str, help="The data of with convai prediction test",
                        default="./input/test_with_convai.csv")
    parser.add_argument("--train_convai_path", type=str, help="The data of with convai prediction train",
                        default="./input/train_with_convai.csv")
    parser.add_argument("--train_indirect_features_path", type=str,
                        help="The file path of train indirect features, if None, we will not import "
                             "the indirect features", default=None)
    parser.add_argument("--mode", type=str, choices=['VALID', 'PREDICT'], help="The mode of train or test", default="VALID")
    parser.add_argument("--save_model_path", type=str, help="The trained model path", default="./trained_models/classifiers")
    parser.add_argument('--pretrained_model_path', type=str, help="The path of pretrained model", default=None)
    parser.add_argument("--tokenizer", type=str, choices=['tokenize', 'tokenize_gensim', 'self_tokenize'],
                        help="The tokenizer of handling words", default="tokenize")
    parser.add_argument("--n_split", type=int, help="The number of split data", default=5)
    parser.add_argument("--word_ngram_range", type=int, nargs="+", help="ngram range of words", default=(1, 2))

    # parser.add_argument("--need_char_features", type=bool, help="Whether need the features of char ngram_range", default=True)
    parser.add_argument('--need_char_features', dest='need_char_features', action='store_true')
    parser.add_argument('--no_char_features', dest='need_char_features', action='store_false')
    parser.set_defaults(need_char_features=True)
    parser.add_argument("--char_ngram_range", type=int, nargs="+", help="ngram range of chars", default=(1, 4))

    # parser.add_argument("--need_count_features", type=bool, help="Whether need the features of char ngram_range",
    #                     default=False)
    parser.add_argument('--need_count_features', dest='need_count_features', action='store_true')
    parser.add_argument('--no_count_features', dest='need_count_features', action='store_false')
    parser.set_defaults(need_count_features=False)

    # parser.add_argument("--need_convai_features", type=bool, help="Whether need the features of convai prediction",
    #                     default=True)
    parser.add_argument('--need_convai_features', dest='need_convai_features', action='store_true')
    parser.add_argument('--no_convai_features', dest='need_convai_features', action='store_false')
    parser.set_defaults(need_convai_features=True)

    parser.add_argument("--count_max_tf", type=float, help="The max term frequency of word with count features",
                        default=0.8)
    parser.add_argument("--count_min_tf", type=float, help="The min term frequency of word with count features",
                        default=0.1)

    parser.add_argument("--result_submission_file", type=str, help="The submission of result",
                        default="./results/submission_nb_svm_tf_idf.csv")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
