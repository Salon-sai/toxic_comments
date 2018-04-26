# -*- coding: utf-8 -*-

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns

train_df = pd.read_csv('./input/train.csv')

labels = train_df.columns[2:].tolist()

suffixes = ["_lr", "_fm_ftrl", "_lgbm", "_nb_svm", "_rnn_cnn_crawl", "_rnn_cnn_glove", "_gru_multipool_crawl",
            "_gru_multipool_glove", "_text_cnn_crawl", "_text_cnn_glove"]
p_lr = pd.read_csv("./results/lr_tf_idf_LB0.9792.csv").rename(
    index=str, columns=dict(zip(labels, [label + "_lr" for label in labels])))

p_fm_ftrl = pd.read_csv('./results/fm_ftrl_submission_LB0.9815.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_fm_ftrl" for label in labels])))


p_lgbm = pd.read_csv('./results/lvl0_lgbm_clean_LB0.9793.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_lgbm" for label in labels])))

p_nb_svm_extend = pd.read_csv('./results/nb_svm_extends_LB0.9781.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_nb_svm" for label in labels])))

# The GRU-multiPooling
p_gru_multipool_glove = pd.read_csv('./results/RNN-multipool/gru_multipool_glove_300_ALL_untrained_epoch_2_LB0.9845.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_gru_multipool_glove" for label in labels])))

p_gru_multipool_crawl = pd.read_csv('./results/RNN-multipool/gru_multipool_crawl_300_ALL_untrained_epoch_3_LB0.9853.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_gru_multipool_crawl" for label in labels])))

# The GRN-CNN
p_gru_cnn_crawl = pd.read_csv('./results/RNN-CNN/conv1d/gru_cnn_crawl_ALL_epoch_3_LB0.9850.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_rnn_cnn_crawl" for label in labels])))

p_gru_cnn_glove = pd.read_csv('./results/RNN-CNN/conv1d/gru_cnn_glove_ALL_epoch_3_LB0.9844.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_rnn_cnn_glove" for label in labels])))

# The GRN-CNN
p_text_cnn_crawl = pd.read_csv('./results/CNN-Text/CNN_TEXT_Crawl_300_ALL_LB0.9825.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_text_cnn_crawl" for label in labels])))

p_text_cnn_glove = pd.read_csv('./results/CNN-Text/CNN_TEXT_GloVe_300_ALL_LB0.9812.csv').rename(
    index=str, columns=dict(zip(labels, [label + "_text_cnn_glove" for label in labels])))


p_final = p_fm_ftrl.merge(p_lr, on="id", how="left")\
    .merge(p_lgbm, on="id", how="left")\
    .merge(p_nb_svm_extend, on="id", how="left")\
    .merge(p_gru_multipool_crawl, on="id", how="left")\
    .merge(p_gru_multipool_glove, on="id", how="left")\
    .merge(p_text_cnn_crawl, on="id", how="left")\
    .merge(p_text_cnn_glove, on="id", how="left")\
    .merge(p_gru_cnn_crawl, on="id", how="left") \
    .merge(p_gru_cnn_glove, on="id", how="left") \

# corr = p_final.iloc[1:].corr()

# Toxic Corr
plt.figure(figsize=(12, 6))
toxic_columns = ["toxic" + suffix for suffix in suffixes]
toxic_corr = p_final[toxic_columns].corr()
plt.subplot(321)
plt.title("Toxic Corr")
sns.heatmap(toxic_corr,
            xticklabels=suffixes,
            yticklabels=suffixes, 
            linewidths=0.05,
            annot=True)

# Severe_toxic Corr
plt.subplot(322)
plt.title("Severe_toxic Corr")
severe_toxic_columns = ["severe_toxic" + suffix for suffix in suffixes]
severe_toxic_corr = p_final[severe_toxic_columns].corr()
sns.heatmap(severe_toxic_corr,
            xticklabels=suffixes,
            yticklabels=suffixes, 
            linewidths=0.05,
            annot=True)

# obscene Corr
plt.subplot(323)
plt.title("obscene Corr")
obscene_columns = ["obscene" + suffix for suffix in suffixes]
obscene_corr = p_final[obscene_columns].corr()
sns.heatmap(obscene_corr,
            xticklabels=suffixes,
            yticklabels=suffixes, 
            linewidths=0.05,
            annot=True)

# threat Corr
plt.subplot(324)
plt.title("threat Corr")
threat_columns = ["threat" + suffix for suffix in suffixes]
threat_corr = p_final[threat_columns].corr()
sns.heatmap(threat_corr,
            xticklabels=suffixes,
            yticklabels=suffixes, 
            linewidths=0.05,
            annot=True)

# insult Corr
plt.subplot(325)
plt.title("insult Corr")
insult_columns = ["insult" + suffix for suffix in suffixes]
insult_corr = p_final[insult_columns].corr()
sns.heatmap(insult_corr,
            xticklabels=suffixes,
            yticklabels=suffixes, 
            linewidths=0.05,
            annot=True)

# identity_hate Corr
plt.subplot(326)
plt.title("identity_hate Corr")
identity_hate_columns = ["identity_hate" + suffix for suffix in suffixes]
identity_hate_corr = p_final[identity_hate_columns].corr()
sns.heatmap(identity_hate_corr,
            xticklabels=suffixes,
            yticklabels=suffixes, 
            linewidths=0.05,
            annot=True)
plt.show()
