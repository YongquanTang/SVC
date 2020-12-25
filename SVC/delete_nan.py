import numpy as np


def delete_nan(np_data, xcol, ycol):
    # Delete rows with nan
    ii = 0
    while ii < np_data.shape[0]:
        if np.isnan(np_data[ii]).any():
            np_data = np.delete(np_data, ii, axis=0)
            continue

        # Delete the row where the water is 0
        k = xcol[len(xcol) - 1]  # k is the last element of xcol, the column where the water is located
        if np_data[ii][k] == 0:
            np_data = np.delete(np_data, ii, axis=0)
            continue
        ii += 1

    X = np_data[:, xcol]
    Y = np_data[:, ycol - 1:ycol]

    return X, Y, np_data
