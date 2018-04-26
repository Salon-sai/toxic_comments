# -*- coding: utf-8 -*-

import os
import re
import string

import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from data_process.APPO import APPO, REPL

nltk.data.path.append("/opt/nltk_data")

eng_stopwords = set(stopwords.words("english"))
tokenizer = TweetTokenizer()
lem = WordNetLemmatizer()

cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

class Indirect_features(object):

    def __init__(self, train_data=None, test_data=None, train_csv_path=None, test_csv_path=None):
        self._train_data = train_data if train_data is not None else pd.read_csv(train_csv_path)
        self._test_data = test_data if test_data is not None else pd.read_csv(test_csv_path)
        self._train_tags = self._train_data.iloc[:, 2:]

        self._all_indirect_features = None
        self._train_indirect_features = None
        self._test_indirect_features = None
        self._spammers = None

        indirect_features_path = '../input/indirect_features.csv'
        if not os.path.exists(indirect_features_path):
            self._generate_indirect_features()
            self._all_indirect_features.to_csv(indirect_features_path, index=False)
        else:
            self._all_indirect_features = pd.read_csv(indirect_features_path)
            self._train_indirect_features = self._all_indirect_features[:len(self._train_data)]
            self._test_indirect_features = self._all_indirect_features[len(self._train_data):]
            self._spammers = self._train_indirect_features[self._train_indirect_features['word_unique_percent'] < 30]

    def _generate_indirect_features(self):
        merge_all = pd.concat([self._train_data.iloc[:, :2], self._test_data.iloc[:, :2]])
        indirect_features = merge_all.reset_index(drop=True)

        bigram = gensim.models.Phrases(indirect_features['comment_text'].values)
        indirect_features['clean_comment_text'] = indirect_features["comment_text"].apply(lambda x: self._clean(x, bigram))

        indirect_features['count_sent'] = indirect_features['comment_text'].apply(
            lambda x: len(re.findall("\n", str(x))) + 1
        )
        indirect_features['count_word'] = indirect_features['clean_comment_text'].apply(
            lambda x: len(str(x).split())
        )
        indirect_features['count_unique_word'] = indirect_features['clean_comment_text'].apply(
            lambda x: len(set(str(x).split()))
        )
        indirect_features['count_letters'] = indirect_features['clean_comment_text'].apply(
            lambda x: len(''.join(str(x).split()))
        )
        indirect_features['count_punctuations'] = indirect_features['comment_text'].apply(
            lambda x: len([w for w in str(x).split() if w in string.punctuation])
        )
        indirect_features['count_words_upper'] = indirect_features['comment_text'].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()])
        )
        indirect_features['count_words_title'] = indirect_features['comment_text'].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()])
        )
        indirect_features['count_stopwords'] = indirect_features["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w in eng_stopwords])
        )
        indirect_features['mean_words_len'] = indirect_features['clean_comment_text'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()])
        )
        indirect_features['num_exclamation_marks'] = indirect_features['comment_text'].apply(lambda x: x.count("!"))
        indirect_features['num_question_marks'] = indirect_features['comment_text'].apply(lambda x: x.count("?"))
        indirect_features['num_smilies'] = indirect_features['comment_text'].apply(
            lambda x: sum(x.count(w) for w in (':-)', ':)', ';-)', ';)')))
        indirect_features['word_unique_percent'] = \
            indirect_features['count_unique_word'] * 100 / indirect_features['count_word']
        indirect_features['punct_percent'] = \
            indirect_features["count_punctuations"] * 100 / indirect_features["count_word"]

        self._all_indirect_features = indirect_features
        self._train_indirect_features = self._all_indirect_features[:len(self._train_data)]
        self._test_indirect_features = self._all_indirect_features[len(self._train_data):]

        self._spammers = self._train_indirect_features[self._train_indirect_features['word_unique_percent'] < 30]

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

def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    df["chick_count"] = df["comment_text"].apply(lambda x: x.count("!"))
    df["qmark_count"] = df["comment_text"].apply(lambda x: x.count("?"))

    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))

def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)

    # 4. Replace some expression with text
    # for expression, word in REPL.items():
    #     clean = clean.replace(bytes(expression, encoding="utf-8"), bytes(word, encoding="utf-8"))

    # 5. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')


def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))
