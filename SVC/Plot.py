import numpy as np
from osgeo import gdal


def plot(auc_score_total, auc_variance_total, n_C, n_gamma, score_path, variance_path):
    print("The output score and variance are grayscale images...")
    # Create image
    image = np.zeros((n_C, n_gamma))
    for i in range(n_C):
        for j in range(n_gamma):
            image[i, j] = auc_score_total[i * n_gamma + j]

    # Output score image
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(score_path, image.shape[1], image.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(image)

    image2 = np.zeros((n_C, n_gamma))
    for i in range(n_C):
        for j in range(n_gamma):
            image2[i, j] = auc_variance_total[i * n_gamma + j]

    # Output variance image
    driver2 = gdal.GetDriverByName("GTiff")
    dataset2 = driver2.Create(variance_path, image2.shape[1], image2.shape[0], 1, gdal.GDT_Float32)
    dataset2.GetRasterBand(1).WriteArray(image2)
    print("OK!")
