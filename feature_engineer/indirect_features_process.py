# -*- coding: utf-8 -*-

import sys
import argparse

import numpy as np
import pandas as pd

from data_process.utils import tokenize
from feature_engineer.indirect_features import Indirect_features

def main(args):
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)

    process = Indirect_features(train_data=train_data, test_data=test_data)
    feature_names = process._all_indirect_features.columns[3:]
    train_data_with_indirect = pd.concat([train_data["id"], process._train_indirect_features[feature_names]], axis=1)
    test_data_with_indirect = pd.concat([test_data["id"], process._test_indirect_features[feature_names]], axis=1)

    train_data_with_indirect.to_csv(args.train_with_indirect_feature_path, index=False)
    test_data_with_indirect.to_csv(args.test_with_indirect_feature_path, index=False)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="The data of test", default="../input/test.csv")
    parser.add_argument("--train_data_path", type=str, help="The data of train", default="../input/train.csv")
    parser.add_argument("--train_with_indirect_feature_path", type=str,
                        help="The train csv of toxic comments with indirect feature",
                        default="../input/train_with_indirect_feature.csv")
    parser.add_argument("--test_with_indirect_feature_path", type=str,
                        help="The test csv of toxic comments with indirect feature",
                        default="../input/test_with_indirect_feature.csv")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

