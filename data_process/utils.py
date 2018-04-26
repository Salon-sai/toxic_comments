# -*- coding: utf-8 -*-

import string, re
import nltk
from collections import defaultdict

from data_process.APPO import APPO, REPL

from gensim.utils import simple_preprocess

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

nltk.data.path.append("/opt/nltk_data")

eng_stopwords = set(stopwords.words("english"))
tokenizer = TweetTokenizer()
lem = WordNetLemmatizer()

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def expression2word(s):
    word_list = str(s).split()
    word_list = [REPL[word] if word in REPL else word for word in word_list]
    word_list = [APPO[word] if word in REPL else word for word in word_list]
    return " ".join(word_list)

def tokenize(s):
    """
    save the punctuations and not changes the lemmatize, it save many words include the number
    """
    words = re_tok.sub(r' \1 ', s).split()
    # words = [repl[word] if word in repl else word for word in words]
    words = [word for word in words if not word in eng_stopwords]
    return words

def tokenize_gensim(s):
    """
    clean the punctuations but not changes the lemmatize
    """
    return simple_preprocess(s, deacc=True, min_len=3)

def self_tokenize(s):
    """
    save the punctuation and change the lemmatize
    """
    s = s.lower()
    s = re.sub("\\n", " ", s)
    s = re.sub("\d{1, 3}.\d{1, 3}.\d{1, 3}.\d{1, 3}", " ", s)
    s = re.sub("\[\[.*\]", " ", s)
    if len(s) == 0:
        s = "unknown"
    words = tokenizer.tokenize(s)
    words = [APPO[word] if word in APPO else word for word in words]

    words = [lem.lemmatize(word, "v") for word in words]
    words = [word for word in words if not word in eng_stopwords]

    return words

def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
