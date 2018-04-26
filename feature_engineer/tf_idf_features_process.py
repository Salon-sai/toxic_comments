# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from data_process.utils import tokenize

class TfidfFeatures(object):

    def __init__(self, comments, train_data=None, test_data=None, train_csv_file=None, test_csv_file=None):
        if comments is None:
            if train_data is None:
                train_data = pd.read_csv(train_csv_file)

            if test_data is None:
                test_data = pd.read_csv(test_csv_file)
            self._comments = train_data['comment_text'].tolist() + test_data['comment_text'].tolist()
        else:
            self._comments = comments

class WordTfidfFeatures(TfidfFeatures):

    def __init__(self, comments, train_data=None, test_data=None, ngram_range=(1, 2),
                 tokenizer=None, train_csv_file=None, test_csv_file=None,
                 max_df=0.9, min_df=3, smooth_idf=1, sublinear_tf=1):
        TfidfFeatures.__init__(self, comments, train_data, test_data, train_csv_file, test_csv_file)

        tokenizer = tokenize if tokenizer is None else tokenizer

        self.word_vec = TfidfVectorizer(max_df=max_df, min_df=min_df, strip_accents='unicode', use_idf=1,
                                        ngram_range=ngram_range, tokenizer=tokenizer, smooth_idf=smooth_idf,
                                        sublinear_tf=sublinear_tf, analyzer="word")
        self.tf_idf_features = self.word_vec.fit_transform(self._comments)


class CharTfidfFeatures(TfidfFeatures):
    def __init__(self, comments, train_data=None, test_data=None, ngram_range=(1, 5),
                 tokenizer=None, train_csv_file=None, test_csv_file=None,
                 max_df=0.9, min_df=0.1, smooth_idf=1, sublinear_tf=1):
        TfidfFeatures.__init__(self, comments, train_data, test_data, train_csv_file, test_csv_file)

        # if comments is None:
        #     if train_data is None:
        #         train_data = pd.read_csv(train_csv_file)
        #
        #     if test_data is None:
        #         test_data = pd.read_csv(test_csv_file)
        #     self._comments = train_data['comment_text'].tolist() + test_data['comment_text'].tolist()
        # else:
        #     self._comments = comments

        tokenizer = tokenize if tokenizer is None else tokenizer

        self._char_vec = TfidfVectorizer(encoding="utf-8", max_df=max_df, min_df=min_df, strip_accents='unicode',
                                           ngram_range=ngram_range, tokenizer=tokenizer, smooth_idf=smooth_idf,
                                           sublinear_tf=sublinear_tf, analyzer="char")
        self.tf_idf_features = self._char_vec.fit_transform(self._comments)

class CountFeatures(TfidfFeatures):
    def __init__(self, comments, train_data=None, test_data=None, ngram_range=(1, 1),
                 tokenizer=None, train_csv_file=None, test_csv_file=None,
                 max_df=0.7, min_df=0.1):
        TfidfFeatures.__init__(self, comments, train_data, test_data, train_csv_file, test_csv_file)

        tokenizer = tokenize if tokenizer is None else tokenizer

        # self._char_vec = TfidfVectorizer(encoding="utf-8", max_df=max_df, min_df=min_df, strip_accents='unicode',
        #                                    ngram_range=ngram_range, tokenizer=tokenizer, smooth_idf=smooth_idf,
        #                                    sublinear_tf=sublinear_tf, analyzer="char")
        self._count_vec = CountVectorizer(max_df=max_df, min_df=min_df, strip_accents='unicode', ngram_range=ngram_range,
                                          tokenizer=tokenizer)

        self.count_features = self._count_vec.fit_transform(self._comments)
