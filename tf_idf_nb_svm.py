# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re, string
import markovify as mk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from model.NB_SVM import NbSvmClassifier
from evaluate import *

from data_process.split_data import *
from data_process.extend_text import extend_text_by_markovify

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
subm = pd.read_csv("./input/sample_submission.csv")
label_cols = train.columns[2:].values
lens = train.comment_text.str.len()
num_samples = train.shape[0]
print(lens.mean(), lens.std(), lens.max(), "\n")
train['none'] = 1 - train[label_cols].max(axis=1)
train_labels = train[label_cols]
print(train_labels.sum())
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
# Building the model
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()
# vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
#                       min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#                       smooth_idf=1, sublinear_tf=1)
# train_tf_idf = vec.fit_transform(train[COMMENT])
# test_tf_idf = vec.transform(test[COMMENT])

# print(train_tf_idf.shape)               # 其shape是(语料数目/样本数目/评论数目 ,所有语料的词语数目)
# print(len(vec.get_feature_names()))     # 获取所有的words

def validate_split_train_data(train_data, train_features, columns_name):
    thresholds = np.arange(0, 1, 0.01)
    auc_array = []
    for i, label in enumerate(columns_name):
        print("validate fit", label)
        # train_indices, valid_indices = split_train_valid(train, label)
        positive_train_indices, positive_valid_indices, negative_train_indices, negative_valid_indices = \
            split_train_valid(train_data, label)

        valid_indices = np.concatenate([positive_valid_indices, negative_valid_indices])
        X_pos_train, X_neg_train, X_valid = train_features[positive_train_indices.values, :], \
                                            train_features[negative_train_indices.values, :], \
                                            train_features[valid_indices, :]
        y_pos_train, y_neg_train, y_valid = train_data[label].iloc[positive_train_indices.values], \
                                            train_data[label].iloc[negative_train_indices.values], \
                                            train_data[label].iloc[valid_indices]
        preds_ps = []
        for X, y in split_imbalanced_data(X_pos_train, X_neg_train, y_pos_train, y_neg_train):
            model = NbSvmClassifier(C=4, dual=True)
            model.fit(X, y)
            preds_ps.append(model.predict_proba(X_valid)[:, 1])
        preds_p = np.asarray(preds_ps).mean(0)
        tprs, fprs, _ = evaluate(preds_p, y_valid.values, thresholds, label)
        auc = metrics.auc(fprs, tprs)
        auc_array.append(auc)
        print("ROC of %s comment, AUC: %1.3f" % (label, auc))

    print("Mean column-wise AUC: %1.4f" % np.mean(auc_array))

def predict_test(test_data, train_data, train_tf_idf, test_tf_idf, columns_name):
    test_preds = np.zeros((len(test_data), len(columns_name)))
    for i, label in enumerate(columns_name):
        print('fit', label)
        positive_indices, negative_indices = split_by_label(train_data, label)
        positive_X, positive_y, negative_X, negative_y = train_tf_idf[positive_indices.values, :], \
                                                         train_data[label].iloc[positive_indices.values], \
                                                         train_tf_idf[negative_indices.values, :], \
                                                         train_data[label].iloc[negative_indices.values]
        preds = []
        for X, y in split_imbalanced_data(positive_X, negative_X, positive_y, negative_y):
            model = NbSvmClassifier(C=4, dual=True)
            model.fit(X, y)
            preds.append(model.predict_proba(test_tf_idf)[:, 1])
        test_preds[:, i] = np.asarray(preds).mean(0)

    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(test_preds, columns=columns_name)], axis=1)
    submission.to_csv('submission_1.csv', index=False)

def use_extend_text_train_predict(train_df, test_df, label_cols):
    test_preds = np.zeros((len(test), len(label_cols)))
    all_train_comments = train_df.comment_text.tolist()
    all_test_comments = test_df.comment_text.tolist()
    stack_prepare = pd.DataFrame({"id": train_df["id"]})
    valid_scores = []
    for i, label in enumerate(label_cols):
        print('fit %s...' % label)
        class_comments_list = train_df[train[label].eq(1)].comment_text
        extend_texts = extend_text_by_markovify(class_comments_list)
        all_texts = all_train_comments + all_test_comments + extend_texts

        vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                              min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                              smooth_idf=1, sublinear_tf=1)

        vec.fit(all_texts)
        train_words_features = vec.transform(all_train_comments)
        print("Validing..", label)
        stack_prepare[label], full_score = valid(train_words_features, train_df[label].values)
        valid_scores.append(full_score)

        test_words_features = vec.transform(all_test_comments)
        print("\tTraining all train data and predicting the test data...")
        model = NbSvmClassifier(C=4, dual=True)
        model.fit(train_words_features, train_df[label])
        test_preds[:, i] = model.predict_proba(test_words_features)[:, 1]
    stack_prepare.to_csv("./stacking_prepare/nb_svm_extends.csv", index=False)
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(test_preds, columns=label_cols)], axis=1)
    submission.to_csv('./results/nb_svm_extends.csv', index=False)

def valid(train_features, train_labels):
    kfold = KFold(n_splits=5)
    class_preds = np.zeros(len(train_labels), dtype=np.float32)
    for fold_index, (train_indices, valid_indices) in enumerate(kfold.split(train_features, train_labels)):
        model = NbSvmClassifier(C=4, dual=True)
        model.fit(train_features[train_indices], train_labels[train_indices])
        class_preds[valid_indices] = model.predict_proba(train_features[valid_indices])[:, 1]
        score = metrics.roc_auc_score(train_labels[valid_indices], class_preds[valid_indices])
        print("\t In the fold: %d, the AUC score is %.8f" % (fold_index, score))
    full_score = metrics.roc_auc_score(train_labels, class_preds)
    print("The full score is %.8f" % full_score)
    return class_preds, full_score

use_extend_text_train_predict(train, test, label_cols)