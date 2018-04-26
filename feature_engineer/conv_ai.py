# -*- coding: utf-8 -*-

import argparse
import sys
import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def join_and_sanitize(cmt, annot):
    df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())
    df = sanitize(df)
    return df

def sanitize(df):
    comment = 'comment' if 'comment' in df else 'comment_text'
    df[comment] = df[comment].str.lower().str.replace('newline_token', ' ')
    df[comment] = df[comment].fillna('erikov')
    return df

def Tfidfize(df, max_vocab = 200000):
    comment = 'comment' if 'comment' in df else 'comment_text'
    tf_idfer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_vocab, use_idf=1,
                               stop_words='english', smooth_idf=1, sublinear_tf=1)
    tf_idf = tf_idfer.fit_transform(df[comment])
    return tf_idf, tf_idfer

def tfidf_and_predict(tf_ider, model, train_df, test_df):
    tf_idf_train = tf_ider.transform(train_df.comment_text)
    tf_idf_test = tf_ider.transform(test_df.comment_text)
    train_scores = model.predict(tf_idf_train)
    test_scores = model.predict(tf_idf_test)
    return train_scores, test_scores

def main(args):
    toxic_cmt = pd.read_table(os.path.expanduser(args.toxic_cmt_path))
    toxic_annot = pd.read_table(os.path.expanduser(args.toxic_annot_path))
    aggr_cmt = pd.read_table(os.path.expanduser(args.agg_cmt_path))
    aggr_annot = pd.read_table(os.path.expanduser(args.agg_annot_path))
    attack_cmt = pd.read_table(os.path.expanduser(args.att_cmt_path))
    attack_annot = pd.read_table(os.path.expanduser(args.att_annot_path))

    train_orig = pd.read_csv(os.path.expanduser(args.train_data_path))
    test_orig = pd.read_csv(os.path.expanduser(args.test_data_path))

    toxic = join_and_sanitize(toxic_cmt, toxic_annot)
    attack = join_and_sanitize(attack_cmt, attack_annot)
    aggression = join_and_sanitize(aggr_cmt, aggr_annot)

    print("Start calculate the tf-idf of conv-ai...")
    X_toxic, tfidfer_toxic = Tfidfize(toxic, 30000)
    y_toxic = toxic['toxicity'].values
    print("The shape of toxic TF-IDF:", X_toxic.shape)
    X_attack, tfidfer_attack = Tfidfize(attack, 30000)
    y_attack = attack['attack'].values
    print("The shape of attack TF-IDF:", X_attack.shape)
    X_aggression, tfidfer_aggression = Tfidfize(aggression, 30000)
    y_aggression = aggression['aggression'].values
    print("The shape of aggression TF-IDF:", X_aggression.shape)

    ridge = Ridge()
    mse_toxic = -cross_val_score(ridge, X_toxic, y_toxic, scoring='neg_mean_squared_error')
    mse_attack = -cross_val_score(ridge, X_attack, y_attack, scoring='neg_mean_squared_error')
    mse_aggression = -cross_val_score(ridge, X_aggression, y_aggression, scoring='neg_mean_squared_error')

    print(mse_toxic.mean(), mse_attack.mean(), mse_aggression.mean())

    print("Start fit the ridge model with conv-ai data...")
    model_toxic = ridge.fit(X_toxic, y_toxic)
    model_attack = ridge.fit(X_attack, y_attack)
    model_aggression = ridge.fit(X_aggression, y_aggression)

    train_orig = sanitize(train_orig)
    test_orig = sanitize(test_orig)

    print("Start to predict the scores of toxic comments")
    toxic_train_scores, toxic_test_scores = tfidf_and_predict(tfidfer_toxic, model_toxic, train_orig, test_orig)
    attack_train_scores, attack_test_scores = tfidf_and_predict(tfidfer_attack, model_attack, train_orig, test_orig)
    aggression_train_scores, aggression_test_scores = tfidf_and_predict(tfidfer_aggression, model_aggression,
                                                                        train_orig, test_orig)

    train_orig['toxic_level'] = toxic_train_scores
    train_orig['attack'] = attack_train_scores
    train_orig['aggression'] = aggression_train_scores

    test_orig['toxic_level'] = toxic_test_scores
    test_orig['attack'] = attack_test_scores
    test_orig['aggression'] = aggression_test_scores

    train_orig.to_csv(os.path.expanduser(args.train_with_convai_path), index=False)
    test_orig.to_csv(os.path.expanduser(args.test_with_convai_path), index=False)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of test", default="./input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of train", default="./input/train.csv")

    parser.add_argument("--toxic_cmt_path", type=str, help="The file path of toxicity annotated comments",
                        default="./input/conversationaidataset/toxicity_annotated_comments.tsv")
    parser.add_argument("--agg_cmt_path", type=str, help="The file path of aggression annotated comments",
                        default="./input/conversationaidataset/aggression_annotated_comments.tsv")
    parser.add_argument("--att_cmt_path", type=str, help="The file path of attack annotated comments",
                        default="./input/conversationaidataset/attack_annotated_comments.tsv")

    parser.add_argument("--toxic_annot_path", type=str, help="The file path of toxicity annotations",
                        default="./input/conversationaidataset/toxicity_annotations.tsv")
    parser.add_argument("--agg_annot_path", type=str, help="The file path of aggression annotations",
                        default="./input/conversationaidataset/aggression_annotations.tsv")
    parser.add_argument("--att_annot_path", type=str, help="The file path of attack annotations",
                        default="./input/conversationaidataset/attack_annotations.tsv")

    parser.add_argument("--train_with_convai_path", type=str, help="The train csv of toxic comments with predict of "
                                                                   "features from conversationai dataset",
                        default="./input/train_with_convai.csv")
    parser.add_argument("--test_with_convai_path", type=str, help="The test csv of toxic comments with predict of "
                                                                  "features from conversationai dataset",
                        default="./input/test_with_convai.csv")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
