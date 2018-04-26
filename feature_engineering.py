# -*- coding: utf-8 -*-

# basics
import pandas as pd
import numpy as np
from pandas import Series

# misc
import gc
import time
import warnings
from scipy.misc import imread

# viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib_venn as venn

# nlp
import string
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import sparse
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer

# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# settings
nltk.data.path.append("/opt/nltk_data")
start_time = time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

from data_process.data_process import DataBase
import draw_picture

train_database = DataBase("./input/train.csv", "./input/test.csv")
train_database.statistics()

draw_picture.draw_labels_distribution(train_database.train_data.iloc[:, 2:].sum())

draw_picture.cov_label_heatmap(train_database.train_data.iloc[:, 2: -1])

draw_picture.wordcloud_with_label(train_database.train_data, train_database.labels)

draw_picture.num_sentences_words(train_database.train_features)

draw_picture.unique_words_plt(train_database.train_features, train_database.spammers)






