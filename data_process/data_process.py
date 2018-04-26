# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
import nltk
import gensim

from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from gensim.utils import simple_preprocess

import re, string

nltk.data.path.append("/opt/nltk_data")

eng_stopwords = set(stopwords.words("english"))
tokenizer = TweetTokenizer()
lem = WordNetLemmatizer()

#https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'd" : "I had",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "didn't": "did not",
    "tryin'":"trying"
}

class DataBase(object):

    def __init__(self, train_csv, test_csv,
                 train_feature_path='./input/train_features.csv',
                 test_feature_path='./input/test_features.csv'):
        self._train_csv = train_csv
        self._train_feature_path = train_feature_path
        self._test_feature_path = test_feature_path

        # load the original train data (comments and tags)
        self._train_data = pd.read_csv(train_csv)
        # load the original test data (comments and tags)
        self._test_data = pd.read_csv(test_csv)
        # because train data has no clean tag, we need the generate the clean tag
        self._generate_clean_label()
        # check all data has null and fill the 'unknown' for null comments
        self._check_null_comments()
        # get the name of columns in train data
        self._columns = list(self._train_data)
        # get the labels of comment (toxic, severe_toxic, obscene ...) in train data
        self._labels = self._columns[2:]
        # get the train tags of each comment, we can get each comment label values
        self._train_tags = self._train_data.iloc[:, 2:]
        # get number of comment by label or class
        self._num_labels = self._train_tags.sum()
        # all data indirect features
        self._data_indirect_features = None
        # train data indirect features
        self._train_indirect_features = None
        # test data indirect features
        self._test_indirect_features = None
        # spammers comments in train data
        self._spammers = None
        self._generate_indirect_features()

        self._leaky_features = None

        self._unigram_tf_idfVectorizer = None
        # unigrams words directory
        self._unigram_features = None
        # unigrams tf-idf train features values
        self._train_unigrams = None
        # unigrams tf-idf test features values
        self._test_unigrams = None
        # top tf-idf value of word with different class or label
        self._unigrams_tf_idf_top_n_per_class = None
        self._calculate_unigrams_tf_idf()

        self._bigrams_tf_idfVectorizer = None
        # bigrams words directory
        self._bigrams_features = None
        # bigrams tf-idf train features values
        self._train_bigrams = None
        # bigrams tf-idf test features values
        self._test_bigrams = None
        # top tf-idf value of word with different class or label
        self._bigrams_tf_idf_top_n_per_class = None
        self._calculate_bigrams_tf_idf()

        self._kf = KFold(n_splits=10)
        self._train_test_index_generator = self._kf.split(self._train_data)

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def labels(self):
        return self._labels

    @property
    def train_features(self):
        return self._train_indirect_features

    @property
    def spammers(self):
        return self._spammers

    @property
    def data_indirect_features(self):
        return self._data_indirect_features

    def _generate_clean_label(self):
        labels = self.train_data.iloc[:, 2:]
        _sum = labels.sum(axis=1)
        self.train_data["clean"] = (_sum == 0)

    def next(self):
        train_indices, test_indices = next(self._train_test_index_generator)
        # return self._train_data[train_indices], self._train_data[test_indices]
        return train_indices, test_indices

    def _check_null_comments(self):
        train_null_check = self._train_data.isnull().sum().sum()
        test_null_check = self._test_data.isnull().sum().sum()

        if train_null_check:
            self._train_data['comment_text'].fillna("unknown", inplace=True)
            print("Has fill the train null comment to 'unknown'")
        else:
            print("There is no null comment in training set")

        if test_null_check:
            self._test_data['comment_text'].fillna("unknown", inplace=True)
            print("Has fill the test null comment to 'unknown'")
        else:
            print("There is no null comment in test set")

    def statistics(self):
        print("\n------------")
        print("The number of train samples: %d, The number of test samples: %d" %
              (self._train_data.shape[0], self._test_data.shape[0]))
        for key, data in self._num_labels.items():
            print("The number of %s : %d" % (key, data))

    def _generate_indirect_features(self):
        if not (os.path.exists(self._train_feature_path) and os.path.exists(self._test_feature_path)):
            merge_all = pd.concat([self._train_data.iloc[:, :2], self._test_data.iloc[:, :2]])
            data_features = merge_all.reset_index(drop=True)

            bigram = gensim.models.Phrases(data_features['comment_text'].values)
            data_features['clean_comment_text'] = data_features["comment_text"].apply(lambda x: self._clean(x, bigram))

            data_features['count_sent'] = data_features["comment_text"].apply(
                lambda x: len(re.findall("\n", str(x))) + 1)
            data_features['count_word'] = data_features["clean_comment_text"].apply(lambda x: len(str(x).split()))
            data_features['count_unique_word'] = data_features["clean_comment_text"].apply(lambda x: len(set(str(x).split())))
            data_features['count_letters'] = data_features["clean_comment_text"].apply(lambda x: len(str(x)))
            data_features['count_punctuations'] = data_features["comment_text"].apply(
                lambda x: len([w for w in str(x).split() if w in string.punctuation]))
            data_features['count_words_upper'] = data_features["comment_text"].apply(
                lambda x: len([w for w in str(x).split() if w.isupper()]))
            data_features['count_words_title'] = data_features["comment_text"].apply(
                lambda x: len([w for w in str(x).split() if w.istitle()]))
            data_features['count_stopwords'] = data_features["comment_text"].apply(
                lambda x: len([w for w in str(x).split() if w in eng_stopwords]))
            data_features['mean_word_len'] = data_features["clean_comment_text"].apply(
                lambda x: np.mean([len(w) for w in str(x).split()]))
            data_features['word_unique_percent'] = \
                data_features['count_unique_word'] * 100 / data_features['count_word']
            data_features['punct_percent'] = \
                data_features["count_punctuations"] * 100 / data_features["count_word"]

            self._data_indirect_features = data_features
            self._train_indirect_features = pd.concat([data_features[:len(self._train_data)], self._train_tags], axis=1)
            self._test_indirect_features = data_features[len(self._train_data):]
            # save the feature to csv
            self._train_indirect_features.to_csv(self._train_feature_path, index=False)
            self._test_indirect_features.to_csv(self._test_feature_path, index=False)
        else:
            self._train_indirect_features = pd.read_csv(self._train_feature_path)
            self._test_indirect_features = pd.read_csv(self._test_feature_path)
            self._data_indirect_features = pd.concat([self._train_indirect_features, self._test_indirect_features])

        self._spammers = self._train_indirect_features[self._train_indirect_features['word_unique_percent'] < 30]

    def generate_leaky_features(self):
        pass

    def _calculate_unigrams_tf_idf(self, need_calculate_top_tf_idf=False):
        # some detailed description of the parameters
        # min_df=150 --- ignore terms that appear lesser than 150 times
        # max_features=10000  --- Create as many words as present in the text corpus
        # changing max_features to 1k for memmory issues
        # analyzer='word'  --- Create features from words (alternatively char can also be used)
        # ngram_range=(1,1)  --- Use only one word at a time (unigrams)
        # strip_accents='unicode' -- removes accents
        # use_idf=1,smooth_idf=1 --- enable IDF
        # sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
        self._unigram_tf_idfVectorizer = TfidfVectorizer(min_df=10, max_features=50000, strip_accents='unicode',
                                                         analyzer='word', ngram_range=(1, 1), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                                         stop_words='english')
        self._unigram_tf_idfVectorizer.fit(self._data_indirect_features['clean_comment_text'])
        # get the words in the clean corpus, it means words directory
        self._unigram_features = np.array(self._unigram_tf_idfVectorizer.get_feature_names())
        self._train_unigrams = self._unigram_tf_idfVectorizer.transform(
            self._data_indirect_features['clean_comment_text'].iloc[:self._train_data.shape[0]]
        )
        self._test_unigrams = self._unigram_tf_idfVectorizer.transform(
            self._data_indirect_features['clean_comment_text'].iloc[self._train_data.shape[0]:]
        )
        if need_calculate_top_tf_idf:
            self._unigrams_tf_idf_top_n_per_class = self.top_features_by_class(feature_data=self._train_unigrams,
                                                                               features=self._unigram_features)

    def _calculate_bigrams_tf_idf(self, need_calculate_top_tf_idf=False):
        self._bigrams_tf_idfVectorizer = TfidfVectorizer(min_df=10, max_features=30000, strip_accents='unicode',
                                                         analyzer='word', ngram_range=(2, 2), use_idf=1, smooth_idf=1,
                                                         sublinear_tf=1,
                                                         stop_words='english')
        self._bigrams_tf_idfVectorizer.fit(self._data_indirect_features['clean_comment_text'])
        # get the words in the clean corpus, it means words directory
        self._bigram_features = np.array(self._bigrams_tf_idfVectorizer.get_feature_names())
        self._train_bigrams = self._bigrams_tf_idfVectorizer.transform(
            self._data_indirect_features['clean_comment_text'].iloc[:self._train_data.shape[0]]
        )
        self._test_bigrams = self._bigrams_tf_idfVectorizer.transform(
            self._data_indirect_features['clean_comment_text'].iloc[self._train_data.shape[0]:]
        )
        if need_calculate_top_tf_idf:
            self._bigrams_tf_idf_top_n_per_class = self.top_features_by_class(feature_data=self._train_bigrams,
                                                                              features=self._bigram_features)

    def top_features_by_class(self, feature_data, features, min_tfidf=0.1, top_n=20):
        dfs = []
        cols = self._train_tags.columns
        for col in cols:
            ids = self._train_tags.index[self._train_tags[col] == 1]
            top_features = self.top_mean_features(feature_data, features, ids, min_tfidf, top_n)
            # top_features.label = col
            dfs.append(top_features)
        return dfs

    def top_mean_features(self, feature_data, features, group_ids, min_tfidf=0.1, top_n=25):
        # the features by class
        unigram_features = feature_data[group_ids].toarray()
        unigram_features[unigram_features < min_tfidf] = 0
        # calculate mean tf-idf values of each words in current class
        tf_idf_means = np.mean(unigram_features, axis=0)
        return self.top_tf_idf_features(tf_idf_means, features, top_n)

    def top_tf_idf_features(self, row, features, top_n=25):
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_features = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_features)
        df.columns = ['feature', 'tf_idf']
        return df

    def _clean(self, comment, bigram):
        comment = comment.lower()
        comment = re.sub("\\n", " ", comment)
        comment = re.sub("\d{1, 3}.\d{1, 3}.\d{1, 3}.\d{1, 3}", " ", comment)
        comment = re.sub("\[\[.*\]", " ", comment)
        comment = ' '.join(simple_preprocess(comment, deacc=True, min_len=3))
        words = tokenizer.tokenize(comment)
        words = [APPO[word] if word in APPO else word for word in words]
        words = [lem.lemmatize(word, "v") for word in words]
        words = [word for word in words if not word in eng_stopwords]
        words = bigram[words]

        comment = " ".join(words)
        if len(comment) == 0:
            comment = "unknown"

        return comment

