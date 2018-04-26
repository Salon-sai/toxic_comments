# -*- coding: utf-8 -*-
import numpy as np

import time
from sklearn.ensemble import RandomForestClassifier

def tf_idf_feature_selection(tf_idf_features, labels):
    print("Before selection features: ", tf_idf_features.shape)
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf.fit(tf_idf_features, labels)
    importance = np.asarray(rf.feature_importances_)
    pos_corr_index = np.where(importance > 0)[0]
    print("Selection operation speed: %1.4f" % (time.time() - start_time))
    return tf_idf_features.tocsc()[:, pos_corr_index].tocsr()


