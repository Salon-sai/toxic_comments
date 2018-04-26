# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from evaluate import calculate_roc

train_df = pd.read_csv('../input/train.csv')

p_lr_tf_idf = pd.read_csv("../stacking_prepare/lr_tf_idf.csv")
p_fm_ftrl = pd.read_csv("../stacking_prepare/fm_ftrl_clean.csv")
p_lgbm = pd.read_csv("../stacking_prepare/lvl0_lgbm_clean_1000_LB0.9792.csv")
p_nb_svm_extend = pd.read_csv("../stacking_prepare/nb_svm_extends.csv")

p_text_cnn_crawl = pd.read_csv('../stacking_prepare/text_cnn_baseline_rnn_crawl-300d-2M.vec_train_predicts.csv')
p_text_cnn_glove = pd.read_csv("../stacking_prepare/text_cnn_baseline_gru_glove.840B.300d.txt_train_predicts.csv")

p_rnn_cnn_glove = pd.read_csv('../stacking_prepare/RNN_Conv1d_gru_glove.840B.300d.txt_train_predicts.csv')
p_rnn_cnn_crawl = pd.read_csv('../stacking_prepare/RNN_Conv1d_gru_crawl-300d-2M.vec_train_predicts.csv')

p_rnn_multipool_glove = pd.read_csv("../stacking_prepare/RNN_multi_pool_gru_crawl-300d-2M.vec_train_predicts.csv")
p_rnn_multipool_crawl = pd.read_csv("../stacking_prepare/RNN_multi_pool_gru_glove.840B.300d.txt_train_predicts.csv")

column_names = train_df.columns[2:].values
actual = train_df[column_names].values

# 0.99124947
predicts = (2 * p_rnn_multipool_crawl[column_names] + p_rnn_cnn_crawl[column_names]+ p_fm_ftrl[column_names]
            + p_nb_svm_extend[column_names] + p_lgbm[column_names]).values / 6

# 0.99139071 (will selection)
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_crawl[column_names]+ p_fm_ftrl[column_names]
#             + p_nb_svm_extend[column_names] + p_lgbm[column_names]).values / 6

# 0.99129586
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_lgbm[column_names]).values / 5

# 0.99110840
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names] +
#             p_lr_tf_idf[column_names]).values / 5

# 0.99085033
# predicts = (2 * p_rnn_multipool_crawl[column_names] + p_rnn_multipool_crawl[column_names] +
#             p_fm_ftrl[column_names] + p_nb_svm_extend[column_names] + p_lgbm[column_names]).values / 6

# 0.99138029 (selection)
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names]
#             + p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_nb_svm_extend[column_names] + p_lgbm[column_names]).values / 7

# 0.99132954
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names]
#             + p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_lr_tf_idf[column_names] + p_lgbm[column_names]).values / 7

# 0.99136597
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names]
#             + p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_nb_svm_extend[column_names] + p_lr_tf_idf[column_names] + p_lgbm[column_names]).values / 8

# 0.99129152
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names]
#             + p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_nb_svm_extend[column_names] + p_lr_tf_idf[column_names] + p_lgbm[column_names]
#             + p_text_cnn_glove[column_names] + p_text_cnn_crawl[column_names]).values / 10

# 0.99130281
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names]
#             + p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_nb_svm_extend[column_names] + p_lr_tf_idf[column_names] + p_lgbm[column_names]
#             + p_text_cnn_glove[column_names]).values / 9

# 0.99136039
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names]
#             + p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names]
#             + p_nb_svm_extend[column_names] + p_lr_tf_idf[column_names] + p_lgbm[column_names]
#             + p_text_cnn_crawl[column_names]).values / 9

# 0.99098982
# predicts = (2 * p_rnn_multipool_crawl[column_names] + 2 * p_rnn_cnn_glove[column_names]
#             + p_fm_ftrl[column_names]
#             + p_lr_tf_idf[column_names] + p_lgbm[column_names]).values / 7

# 0.99137614 (selection)
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             + 2 * p_fm_ftrl[column_names] + p_nb_svm_extend[column_names]
#             + p_lgbm[column_names]).values / 8

# 0.99134637
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             + 2 * p_fm_ftrl[column_names] + 2 * p_nb_svm_extend[column_names]
#             + p_lgbm[column_names]).values / 9

# 0.99134497
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             + 2 * p_fm_ftrl[column_names] + p_nb_svm_extend[column_names] + p_lr_tf_idf[column_names]
#             + p_lgbm[column_names]).values / 9

# 0.99131544
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             + 2 * p_fm_ftrl[column_names] + p_lr_tf_idf[column_names]
#             + p_lgbm[column_names]).values / 8

# 0.99124074
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             + 2 * p_fm_ftrl[column_names] + 2 * p_lr_tf_idf[column_names]
#             + p_lgbm[column_names]).values / 9

# 0.99135449
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             2 * p_fm_ftrl[column_names] + p_nb_svm_extend[column_names] +
#             2 * p_lgbm[column_names]).values / 9

# 0.99138754
# predicts = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#             p_rnn_cnn_glove[column_names] + p_rnn_cnn_crawl[column_names] +
#             1.5 * p_fm_ftrl[column_names] + p_nb_svm_extend[column_names] +
#             1.5 * p_lgbm[column_names]).values / 8

scores = calculate_roc(actual, predicts, column_names)
print("Avg AUC score %.8f" % (np.mean(list(scores.values()))))