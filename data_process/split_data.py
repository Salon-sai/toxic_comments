# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import scipy.sparse as sp

from sklearn.model_selection import train_test_split

def split_by_label(data, label_name):
    """
    split the data to positive and negative data
    :param data: data frame data of
    :param label_name: target label name
    :return:
    """
    data = pd.DataFrame(data)
    x = data[label_name].eq(1)
    return x[x].index, x[~x].index

def split_train_valid(data_frame, label_name, need_concatenate=False):
    positive_indices, negative_indices = split_by_label(data_frame, label_name)
    positive_train_indices, positive_valid_indices = train_test_split(positive_indices, test_size=0.3, random_state=2018)
    negative_train_indices, negative_valid_indices = train_test_split(negative_indices, test_size=0.3, random_state=2018)
    if need_concatenate:
        train_indices = np.concatenate([positive_train_indices, negative_train_indices])
        valid_indices = np.concatenate([positive_valid_indices, negative_valid_indices])
        return train_indices, valid_indices
    else:
        return positive_train_indices, positive_valid_indices, negative_train_indices, negative_valid_indices

def split_imbalanced_data(positive_X, negative_X, positive_y, negative_y):
    num_positive = positive_X.shape[0]
    num_negative = negative_X.shape[0]
    if num_negative > num_positive:
        large_samples, large_labels = negative_X, negative_y
        small_samples, small_labels = positive_X, positive_y
    else:
        large_samples, large_labels = positive_X, positive_y
        small_samples, small_labels = negative_X, negative_y

    for i in range(int(np.ceil(large_samples.shape[0] / small_samples.shape[0]))):
        start = i * small_samples.shape[0]
        end = min((i + 1) * small_samples.shape[0], large_samples.shape[0])
        yield sp.vstack([small_samples, large_samples[start: end]]), pd.concat([small_labels, large_labels[start: end]])


