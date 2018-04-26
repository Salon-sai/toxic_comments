# -*- coding: utf-8 -*-

import numpy as np
import time

from feature_engineer.indirect_features import Indirect_features
from model.NB_SVM import NbSvmClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

start_time = time.time()
indirect_features = Indirect_features(
    train_csv_path="./input/train.csv",
    test_csv_path="./input/test.csv")
print("calculate indirect features speed: %4f" % (time.time() - start_time))
SELECTED_COLS = ['count_sent', 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations',
                 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_words_len', 'word_unique_percent',
                 'punct_percent']

train_indirect_features = indirect_features._train_indirect_features
test_indirect_features = indirect_features._test_indirect_features
train_labels = indirect_features._train_tags

X_train, X_valid, y_train, y_valid = train_test_split(
    train_indirect_features[SELECTED_COLS], train_labels, test_size=0.2, random_state=2018)

valid_loss = np.zeros(len(train_labels.columns))

preds_train = np.zeros((X_train.shape[0], len(train_labels.columns)))
preds_valid = np.zeros((X_valid.shape[0], len(train_labels.columns)))
acc_valid = np.zeros(len(train_labels.columns))

for i, label in enumerate(train_labels.columns.values):
    print("Class:= " + label)
    result = np.zeros(len(X_valid))
    model = NbSvmClassifier(C=4, dual=True)
    model.fit(X_train, y_train[label])
    preds_valid[:, i] = model.predict_proba(X_valid)[:, 1]
    preds_train[:, i] = model.predict_proba(X_train)[:, 1]
    valid_loss_class = log_loss(y_valid[label], preds_valid[:, i])
    result[np.where(preds_valid[:, i] > 0.8)[0]] = 1
    accuracy = np.equal(result, y_valid[label]).sum() / X_train.shape[0]
    valid_loss[i] = (valid_loss_class)
    print("valid loss: %1.5f, Accuracy: %1.5f" % (valid_loss_class, accuracy))

preds_result = np.zeros((X_valid.shape[0], len(train_labels.columns)))
preds_result[np.where(preds_valid > 0.8)] = 1
total_accuracy = np.equal(preds_result, preds_valid).sum() / (X_valid.shape[0] * X_valid.shape[1])

print("Total Accuracy: %1.5f" % total_accuracy)
print('mean column-wise log loss:Validation dataset', np.mean(valid_loss))

