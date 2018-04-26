import os
import sys
import argparse
import gc

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def main(args):
    assert (len(args.stacking_files) > 0)
    assert (len(args.stacking_files) == len(args.stacking_results))
    gc.enable()
    train_df = pd.read_csv(args.train_data_path)
    test_df = pd.read_csv(args.test_data_path)
    labels = train_df.columns[2:]
    train_target = train_df[labels]

    final_train_df = merge_csv(labels, args.stacking_files)
    final_test_df = merge_csv(labels, args.stacking_results)

    assert (final_train_df.shape == final_test_df.shape)

    final_columns = final_train_df.columns
    indices = np.arange(train_df.shape[0])
    test_avg_preds = np.zeros((test_df.shape[0], len(labels)))
    valid_preds = np.zeros((train_df.shape[0], len(labels)))
    k_fold = KFold(args.n_split)
    for label_index, label in enumerate(labels):
        print("Valid the %s" % label)
        target = train_target[label].values
        _columns = [column for column in final_columns if column.find(label + "_") >= 0]
        train_features = final_train_df[_columns].values
        test_features = final_test_df[_columns].values
        scores = np.zeros(k_fold.n_splits)
        for n_fold, (train_idx, valid_idx) in enumerate(k_fold.split(indices)):
            model = LogisticRegression()
            # model = RandomForestClassifier(n_estimators=20)
            model.fit(train_features[train_idx], target[train_idx])
            valid_preds[valid_idx, label_index] = model.predict_proba(train_features[valid_idx])[:, 1]
            test_avg_preds[:, label_index] += model.predict_proba(test_features)[:, 1]
            score = roc_auc_score(target[valid_idx], valid_preds[valid_idx, label_index])
            scores[n_fold] = score
            print("\tfold %d, score %.8f" % (n_fold, score))
        full_score = roc_auc_score(target, valid_preds[:, label_index])
        print("\tFull AUC: %.8f, Avg AUC: %.8f" % (full_score, scores.mean()))
    test_avg_preds /= k_fold.n_splits

    test_preds = np.zeros((test_df.shape[0], len(labels)))
    for label_index, label in enumerate(labels):
        print("Full training the %s" % label)
        target = train_target[label].values
        _columns = [column for column in final_columns if column.find(label + "_") >= 0]
        train_features = final_train_df[_columns].values
        test_features = final_test_df[_columns].values
        model = LogisticRegression()
        model.fit(train_features, target)
        test_preds[:, label_index] = model.predict_proba(test_features)[:, 1]

    # submission_id = pd.DataFrame({"id": train_df["id"]})
    # valid_preds_submission = pd.concat([submission_id, pd.DataFrame({labels: valid_preds})], axis=1)
    submission_id = pd.DataFrame({"id": test_df["id"]})
    test_avg_submission = pd.concat([submission_id, pd.DataFrame({labels: test_avg_preds})], axis=1)
    test_submission = pd.concat([submission_id, pd.DataFrame({labels: test_preds})], axis=1)
    test_avg_submission.to_csv(os.path.join("./results", "%s_%d_fold_avg.csv" % 
                                (args.submission_file_basename, k_fold.n_splits)), index=False)
    test_submission.to_csv(os.path.join('./results', "%s_final.csv" % args.submission_file_basename), index=False)

def merge_csv(labels, *stacking_files):
    final_df = pd.DataFrame()
    for index, stacking_file in enumerate(stacking_files):
        print("concat %d csv file" % index)
        csv_data = pd.read_csv(os.path.join("./stacking_prepare", stacking_file))
        rename_labels = [label + "_" + str(index) for label in labels]
        csv_data = csv_data.rename(index=str, columns=dict(zip(labels, rename_labels)))
        if index == 0:
            final_df = csv_data.copy()
        else:
            final_df = final_df.merge(csv_data, on="id", how="left")
    return final_df

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="The data of original train", default="./input/train.csv")
    parser.add_argument("--test_data_path", type=str, help="The data of test", default="./input/test.csv")
    parser.add_argument("--n_split", type=int, help="The number of split data", default=10)
    parser.add_argument("--stacking_files", type=str, nargs="+",
                        help="The stacking submission of training with different models")
    parser.add_argument("--stacking_results", type=str, nargs="+", 
                        help="The test prediction submission of different models")
    parser.add_argument("--submission_file_basename", type=str, help="second_stacking", default="The file basename of submission file")
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
