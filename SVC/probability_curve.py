import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


def probability_curve(hist_csv_path, hist_pic_path):
    # Read the csv
    pd_curve_data = pd.read_csv(hist_csv_path)
    np_curve_data = pd_curve_data.values
    # Computing Center
    x_sum = [0 for i in range(np_curve_data.shape[1])]
    x_mean = [0 for i in range(np_curve_data.shape[1])]
    # Calculate the sum of each column
    for i in range(np_curve_data.shape[0]):
        for j in range(np_curve_data.shape[1]):
            x_sum[j] += np_curve_data[i][j]

    # Calculate the average of each column
    for k in range(np_curve_data.shape[1]):
        x_mean[k] = x_sum[k] / np_curve_data.shape[0]

    # Calculate the distance from each point to the center of the class
    x_distance = [0 for i in range(np_curve_data.shape[0])]
    d_sum = 0

    for i in range(np_curve_data.shape[0]):
        for j in range(np_curve_data.shape[1]):
            d_sum += ds(np_curve_data[i][j], x_mean[j])
        x_distance[i] = sqrt(d_sum)
        d_sum = 0

    draw_hist(x_distance, hist_pic_path)


# Calculate the square of the difference
def ds(x, y):
    return (x - y) * (x - y)


# What lenths accepts is actually the array passed by sizeArry, which is the data returned by def get_data(lines)
def draw_hist(lenths, hist_pic_path):
    data = lenths
    # Slice the data, divide the data into 20 groups from the minimum to the maximum
    bins = np.linspace(min(data), max(data), 30)
    # This is to call the function to draw the histogram, which means to draw the data according to the division of bins
    plt.hist(data, bins)
    # Set the abscissa
    plt.xlabel('Distance from the data point to the class center')
    # Set the title of the ordinate
    plt.ylabel('Number of data points')
    # Set the title of the entire image
    plt.title('Frequency distribution of data')
    # Save the picture
    plt.savefig(hist_pic_path, format='png')
    # Show the picture
    plt.show()
