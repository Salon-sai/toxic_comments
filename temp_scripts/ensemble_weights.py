import pandas as pd

p_gru_multipool_glove = pd.read_csv('./results/RNN-multipool/gru_multipool_glove_300_ALL_untrained_LB0.9845.csv')

p_gru_multipool_crawl = pd.read_csv('./results/RNN-multipool/gru_multipool_crawl_300_ALL_untrained_epoch_3_LB0.9853.csv')

p_fm_ftlr = pd.read_csv('./results/fm_ftrl_submission_LB0.9815.csv')

p_lgbm = pd.read_csv('./results/lvl0_lgbm_clean_LB0.9793.csv')

p_gru_cnn_crawl = pd.read_csv('./results/RNN-CNN/conv1d/gru_cnn_crawl_ALL_LB0.9847.csv')

sum = 0.9845 + 0.9849 + 0.9815 + 0.9793 + 0.9847

column_names = p_fm_ftlr.columns[1:].values

p_result = p_lgbm.copy()

p_result[column_names] = (0.9845 *p_gru_multipool_glove[column_names] + 0.9849 * p_gru_multipool_crawl[column_names] +
                          0.9815 * p_fm_ftlr[column_names] + 0.9793 * p_lgbm[column_names] +
                          0.9847 * p_gru_cnn_crawl[column_names]) / sum

p_result.to_csv("./results/final_ensemble.csv", index=False)
