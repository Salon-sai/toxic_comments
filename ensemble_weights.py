# -*- coding: utf-8 -*-

import pandas as pd

p_lr = pd.read_csv('./results/lr_tf_idf_LB0.9792.csv')
p_nb_svm_extend = pd.read_csv('./results/nb_svm_extends_LB0.9781.csv')
p_lgbm_500 = pd.read_csv('./results/lvl0_lgbm_clean_LB0.9793.csv')
p_lgbm_1000 = pd.read_csv("./results/lvl0_lgbm_clean_1000_LB0.9792.csv")
p_fm_ftrl = pd.read_csv('./results/fm_ftrl_submission_LB0.9815.csv')

p_cnn_text_crawl = pd.read_csv('./results/CNN-Text/CNN_TEXT_Crawl_300_ALL_LB0.9825.csv')
p_cnn_text_glove = pd.read_csv('./results/CNN-Text/CNN_TEXT_GloVe_300_ALL_LB0.9812.csv')

p_cnn_text_crawl_avg = pd.read_csv("./results/CNN-Text/CNN_TEXT_Crawl_300_STACK_5_fold_avg_LB0.9832.csv")
p_cnn_text_glove_avg = pd.read_csv("./results/CNN-Text/CNN_TEXT_GloVe_300_STACK_5_fold_avg_LB0.9827.csv")

p_rnn_cnn_crawl = pd.read_csv('./results/RNN-CNN/conv1d/gru_cnn_crawl_ALL_epoch_3_LB0.9850.csv')
p_rnn_cnn_glove = pd.read_csv('./results/RNN-CNN/conv1d/gru_cnn_glove_ALL_epoch_3_LB0.9844.csv')

p_rnn_multipool_crawl = pd.read_csv('./results/RNN-multipool/gru_multipool_crawl_300_ALL_untrained_epoch_3_LB0.9853.csv')
p_rnn_multipool_glove = pd.read_csv('./results/RNN-multipool/gru_multipool_glove_300_ALL_untrained_epoch_2_LB0.9845.csv')

p_rnn_multipool_crawl_avg = pd.read_csv("./results/RNN-multipool/RNN_multipool_Crawl_300_STACK_5_fold_avg_LB0.9852.csv")
p_rnn_multipool_glove_avg = pd.read_csv("./results/RNN-multipool/RNN_multipool_GloVe_300_STACK_5_fold_avg_LB0.9857.csv")

column_names = p_lr.columns[1:].values

p_result = p_lr.copy()
# p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#                           p_rnn_cnn_crawl[column_names] + p_fm_ftrl[column_names] + p_lgbm_500[column_names]) / 5

# 0.9871
# p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#                           p_rnn_cnn_crawl[column_names] + p_rnn_cnn_glove[column_names] + p_fm_ftrl[column_names] +
#                           p_nb_svm_extend[column_names] + p_lgbm_500[column_names]) / 7

# 0.9870
# p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#                           p_rnn_cnn_crawl[column_names] + p_rnn_cnn_glove[column_names] + p_fm_ftrl[column_names] +
#                           p_nb_svm_extend[column_names] + p_lgbm_1000[column_names]) / 7

# 0.9871
# p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#                           p_rnn_cnn_crawl[column_names] + p_rnn_cnn_glove[column_names] +
#                           2 * p_fm_ftrl[column_names] + p_nb_svm_extend[column_names] + p_lgbm_500[column_names]) / 8

# 0.9871
# p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#                           p_rnn_cnn_crawl[column_names] + p_rnn_cnn_glove[column_names] +
#                           2 * p_fm_ftrl[column_names] + p_nb_svm_extend[column_names] + p_lgbm_1000[column_names]) / 8

p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
                          p_rnn_cnn_crawl[column_names]+ p_fm_ftrl[column_names] +
                          p_nb_svm_extend[column_names] + p_lgbm_1000[column_names]).values / 6

# p_result[column_names] = (p_rnn_multipool_crawl[column_names] + p_rnn_multipool_glove[column_names] +
#                           p_rnn_cnn_crawl[column_names] + p_rnn_cnn_glove[column_names] +
#                           1.5 * p_fm_ftrl[column_names] + 1.5 * p_nb_svm_extend[column_names] + p_lgbm_500[column_names]) / 8

p_result.to_csv('./results/final_ensemble.csv', index=False)

# p_result['toxic'] = (p_rnn_multipool_glove['toxic'] + p_rnn_cnn_crawl['toxic'] + p_fm_ftrl['toxic']
#                      + p_cnn_text_crawl['toxic']) / 4

# p_result['severe_toxic'] = (p_rnn_cnn_glove['severe_toxic'] + p_rnn_multipool_crawl['severe_toxic'] + p_fm_ftrl['severe_toxic']
#                             + p_cnn_text_crawl['severe_toxic']) / 4

# p_result['obscene'] = (p_fm_ftrl['obscene'] + p_rnn_cnn_crawl['obscene'] + p_rnn_multipool_crawl['obscene']
#                        + p_cnn_text_crawl['obscene']) / 4

# p_result['threat'] = (p_lr['threat'] + p_fm_ftrl['threat'] + p_nb_svm_extend['threat'] + p_cnn_text_crawl['threat']
#                       + p_rnn_cnn_glove['threat']) / 4

# p_result['insult'] = (p_rnn_multipool_glove['insult'] + p_rnn_cnn_glove['insult'] + p_cnn_text_crawl['insult']
#                       + p_fm_ftrl['insult'] + p_lr['insult']) / 4

# p_result['identity_hate'] = (p_rnn_multipool_glove['identity_hate'] + p_rnn_cnn_glove['identity_hate']
#                              + p_cnn_text_crawl['identity_hate'] + p_fm_ftrl['identity_hate'] + p_lr['identity_hate']) / 4

# p_result.to_csv('./results/final_ensemble.csv', index=False)


