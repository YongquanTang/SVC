import os
import sys
import configparser
import numpy as np
from choose_sets import choose_sets
from train import train
from normal import normal
from probability_curve import probability_curve


def main(hist_csv_path, hist_pic_path, paramtxtpath, to_normalized_path, normalized_out_path, stretch, max_array,
         min_array, filepath, trainpath, testpath, testproportion, xcol, ycol, score_path, variance_path, path):
    # Draw data histogram
    if if_hist:
        probability_curve(hist_csv_path, hist_pic_path)
        return
    # Open txt
    fw = open(paramtxtpath, 'w+')

    # Normalized
    if normalized:
        normal(to_normalized_path, normalized_out_path, stretch, max_array, min_array)
        return

    # Partition data set
    choose_sets(filepath, trainpath, testpath, testproportion)

    # Training
    train(filepath, xcol, ycol, param_grid, fw, n_C, n_gamma, w_balanced, w_positive, w_negative, positive, negative,
          score_path, variance_path, n_splits, path)


if __name__ == '__main__':
    # -----------------------------------------Get configuration file parameters---------------------------------------
    cp = configparser.ConfigParser()
    cp.read("config.ini", encoding='utf-8-sig')

    path = cp.get("parameters", "path")
    test_proportion = cp.getfloat("parameters", "test_proportion")
    x_col = eval(cp.get("parameters", "x_col"))
    y_col = cp.getint("parameters", "y_col")

    normalized = cp.getboolean("parameters", "normalized")
    to_normalized_path = cp.get("parameters", "to_normalized_path")
    normalized_out_path = cp.get("parameters", "normalized_out_path")
    stretch = cp.getfloat("parameters", "stretch")
    max_array = eval(cp.get("parameters", "max_array"))
    min_array = eval(cp.get("parameters", "min_array"))

    param_grid = eval(cp.get("parameters", "param_grid"))
    n_splits = cp.getint("parameters", "n_splits")
    n_C = cp.getint("parameters", "n_C")
    n_gamma = cp.getint("parameters", "n_gamma")
    w_balanced = cp.getboolean("parameters", "w_balanced")

    w_positive = cp.getint("parameters", "w_positive")
    w_negative = cp.getint("parameters", "w_negative")

    positive = cp.getint("parameters", "positive")
    negative = cp.getint("parameters", "negative")

    if_hist = cp.getboolean("parameters", "if_hist")

    # -----------------------------------------------Call the main function--------------------------------------
    main(path + 'hist.csv', path + 'hist.png', path + 'param.txt', to_normalized_path, normalized_out_path, stretch,
         max_array, min_array, path + 'complete.csv', path + 'train.csv', path + 'test.csv', test_proportion,
         x_col, y_col, path + 'score.tif', path + 'variance.tif', path)
