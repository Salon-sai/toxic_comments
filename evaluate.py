# -*- coding: utf-8 -*-

import numpy as np

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback

import matplotlib.pyplot as plt

def calculate_log_loss(predicts, actuals, column_names):
    log_losses = []
    thresholds = np.arange(0, 1, 0.01)
    i = 1
    plt.figure(figsize=(6, 4))
    for column_name, predict, actual in zip(column_names, predicts, actuals):
        class_log_loss = metrics.log_loss(actual, predict)
        log_losses.append(class_log_loss)
        tprs, fprs, best_threshold = evaluate(predict, actual, thresholds, column_name)
        auc = metrics.auc(fprs, tprs)
        plt.subplot(3, 2, i)
        plt.plot(fprs, tprs)
        plt.title("ROC of %s comment, AUC: %1.3f" % (column_name, auc), fontsize=15)
        print("Class: %s Log loss: %1.4f" % (column_name, class_log_loss))
        i += 1
    print("Mean column-wise log loss: %1.4f" % np.mean(log_losses))
    plt.show()
    return log_losses

def evaluate(predict, actual, thresholds, column_name, num_folds=10):
    num_thresholds = len(thresholds)
    num_pos = np.sum(actual, dtype=np.int)
    num_neg = np.sum(np.logical_not(actual), dtype=np.int)

    accuracies = np.zeros(num_thresholds)
    tprs = np.zeros(num_thresholds)
    fprs = np.zeros(num_thresholds)
    vals = np.zeros(num_thresholds)
    fars = np.zeros(num_thresholds)
    for i, threshold in enumerate(thresholds):
        tprs[i], fprs[i], accuracies[i] = calculate_accuracy(threshold, predict, actual)
        vals[i], fars[i] = calculate_var_far(threshold, predict, actual, num_pos, num_neg)
    print("\nClass:= ", column_name)
    print("Best Accuracy: %1.4f, number of positive: %d, number of negative: %d" % (accuracies.max(), num_pos, num_neg))
    print("Accuracy: %1.4f+-%1.4f" % (accuracies.mean(), accuracies.std()))
    print("Validation rate: %2.5f+-%2.5f, FAR: %2.5f\n" % (vals.mean(), vals.std(), fars.mean()))
    return tprs, fprs, thresholds[np.argmax(accuracies)]

def calculate_accuracy(threshold, predicts, actual):
    predict_result = np.greater(predicts, threshold)
    predict_result.astype(np.int)
    tp = np.sum(np.logical_and(predict_result, actual))
    fp = np.sum(np.logical_and(predict_result, np.logical_not(actual)))
    tn = np.sum(np.logical_and(np.logical_not(predict_result), np.logical_not(actual)))
    fn = np.sum(np.logical_and(np.logical_not(predict_result), actual))

    tpr = 0 if tp + fn == 0 else tp / float(tp + fn)
    fpr = 0 if fp + tn == 0 else fp / float(fp + tn)
    acc = float(tp + tn) / len(predict_result)

    return tpr, fpr, acc

def calculate_var_far(threshold, predicts, actual, num_positive=None, num_negative=None):
    predict_result = np.greater(predicts, threshold)
    predict_result.astype(np.int)
    true_accept = np.sum(np.logical_and(predict_result, actual))
    false_accept = np.sum(np.logical_and(predict_result, np.logical_not(actual)))
    num_positive = num_positive or np.sum(actual)
    num_negative = num_negative or np.sum(np.logical_not(actual))
    val = float(true_accept) / num_positive
    far = float(false_accept) / num_negative
    return val, far

def calculate_roc(actuals, predicts, column_names):
    assert (actuals.shape == predicts.shape)
    assert (predicts.shape[1] == len(column_names))
    scores = dict()
    for column_index, column_name in enumerate(column_names):
        actual = actuals[:, column_index]
        predict = predicts[:, column_index]
        score = roc_auc_score(actual, predict)
        print("The AUC score of %s: %.8f" % (column_name, score))
        scores[column_name] = score
    return scores

class RocAucCheckpoint(Callback):

    def __init__(self, file_path, validation_data=(), interval=1, save_weights_only=True, verbose=1,
                 valid_pred_file=None):
        super(Callback, self).__init__()

        self.file_path = file_path
        self._interval = interval
        self._X_val, self._y_val = validation_data
        self.best_auc = -np.Inf
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        # self.valid_pred_file = valid_pred_file

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._interval == 0:
            y_pre = self.model.predict(self._X_val, verbose=0)
            score = roc_auc_score(self._y_val, y_pre)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))
            if score >= self.best_auc:
                file_path = self.file_path + "_{score}".format(score=score)
                if self.verbose > 0:
                    print('Epoch %05d: AUC improved from %0.5f to %0.5f,'
                          ' saving model to %s' % (epoch, self.best_auc, score, file_path))
                self.best_auc = score
                if self.save_weights_only:
                    self.model.save_weights(file_path, overwrite=True)
                else:
                    self.model.save(file_path, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: AUC did not improve' % (epoch))

    # def save_valid_predict(self, y_predict):
    #     save_path = "./stacking_prepare"
    #     if self.valid_pred_file is None:
    #         return
    #