# -*- coding: utf-8 -*-

import markovify as mk

def extend_text_by_markovify(class_comments_list):
    mkv_text = []
    if len(class_comments_list) < 7500:
        text_model = mk.Text(class_comments_list.values)
        nchar = int(class_comments_list.str.len().median())
        for _ in range(len(class_comments_list) // 2):
            new_text = text_model.make_short_sentence(nchar)
            mkv_text.append(new_text)
    return mkv_text

# TODO: extend the data with RNN
def extend_text_by_rnn(model_path, label):
    pass

def build_rnn(class_comment_list, label, model_path):
    pass