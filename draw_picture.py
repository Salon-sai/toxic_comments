# -*- coding: utf-8 -*-

import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from wordcloud import WordCloud, STOPWORDS

def draw_labels_distribution(num_labels, save_fig_path="./pic/distribution.png"):
    if not os.path.exists(save_fig_path):
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(num_labels.index, num_labels.values, alpha=0.8)
        plt.title("Per class")
        plt.ylabel("# of occurrences", fontsize=12)
        plt.xlabel("Type", fontsize=12)

        rects = ax.patches
        labels = num_labels.values

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom")

        plt.savefig(fname=save_fig_path)

def cov_label_heatmap(labels_data, save_fig_path="./pic/covariance.png"):
    if not os.path.exists(save_fig_path):
        corr = labels_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, linewidths=1)
        plt.savefig(fname=save_fig_path)

def wordcloud_with_label(train_data, labels, save_fig_path="./pic/word_cloud.png"):
    if not os.path.exists(save_fig_path):
        stopword = set(STOPWORDS)
        plt.figure(figsize=(100, 100))
        for i, label in enumerate(labels):
            plt.subplot2grid((3, 3), (i // 3, i % 3))
            plt.axis("off")
            plt.title("Frequent in %s" % label, fontsize=20)

            subset = train_data[train_data[label] == 1]
            text = subset.comment_text.values
            word_cloud = WordCloud(background_color="black", max_words=400, margin=2, stopwords=stopword)
            word_cloud.generate(" ".join(text))
            plt.imshow(word_cloud.recolor(random_state=244), alpha=0.98)

        plt.savefig(fname=save_fig_path)

def num_sentences_words(train_features, save_fig_path="./pic/num_sentences.png"):
    if not os.path.exists(save_fig_path):
        plt.figure(figsize=(12, 6))

        train_features['count_sent'].loc[train_features['count_sent'] > 10] = 10
        train_features['count_word'].loc[train_features['count_word'] > 200] = 200

        # sentences
        plt.subplot(121)
        plt.suptitle("Are longer comments more toxic?", fontsize=20)
        sns.violinplot(y="count_sent", x='clean', data=train_features, split=True)
        plt.xlabel("Clean?", fontsize=12)
        plt.ylabel("of sentences", fontsize=12)
        plt.title("Number of sentences in each comment", fontsize=15)

        # words
        train_features['count_word'].loc[train_features['count_word'] > 200] = 200
        plt.subplot(122)
        sns.violinplot(y='count_word', x='clean', data=train_features, split=True, inner="quart")
        plt.xlabel("Clean?", fontsize=12)
        plt.ylabel("# words", fontsize=12)
        plt.title("Number of words in each comment", fontsize=15)
        plt.savefig(fname=save_fig_path)


def unique_words_plt(train_features, spammers, save_fig_path="./pic/unique_words.png"):
    if not os.path.exists(save_fig_path):
        train_features['count_unique_word'].loc[train_features['count_unique_word'] > 200] = 200
        temp_data_features = pd.melt(train_features, value_vars=['count_word', 'count_unique_word'], id_vars='clean')
        plt.figure(figsize=(16, 12))
        plt.suptitle("What's so unique?", fontsize=20)
        gridspec.GridSpec(2, 2)
        plt.subplot2grid((2, 2), (0, 0))
        sns.violinplot(x="variable", y="value", hue="clean", data=temp_data_features, split=True, inner="quartile")
        plt.title("Absolute wordcount and unique words count")
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Count", fontsize=12)

        plt.subplot2grid((2, 2), (0, 1))
        plt.title("Percentage of unique words of total words in comment")
        ax = sns.kdeplot(train_features[train_features.clean == 0].word_unique_percent, label="Bad", shade=True, color='r')
        ax = sns.kdeplot(train_features[train_features.clean == 1].word_unique_percent, label="Clean")

        plt.legend()
        plt.ylabel("Number of occurances", fontsize=12)
        plt.xlabel("Percent unique words", fontsize=12)

        plt.subplot2grid((2, 2), (1, 0), colspan=2)
        spammer_labels = spammers.iloc[:, -7:].sum()
        plt.title("Count of comments with low(<30%) unique words", fontsize=15)
        ax = sns.barplot(x=spammer_labels.index, y=spammer_labels.values)

        rects = ax.patches
        labels = spammer_labels.values

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
        plt.xlabel('Threat class', fontsize=12)
        plt.xlabel('# of comments', fontsize=12)

        plt.savefig(fname=save_fig_path)