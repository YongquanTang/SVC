import pandas as pd


def normal(to_normal_path, normal_result_path, stretch, max_array, min_array):
    # Read the csv
    pd_normal_data = pd.read_csv(to_normal_path)
    np_normal_data = pd_normal_data.values

    # Normalized
    for i in range(np_normal_data.shape[0]):
        for j in range(len(max_array)):
            np_normal_data[i][j + 2] = normal1(np_normal_data[i][j + 2], max_array[j], min_array[j],
                                               stretch)
    # Delete original data
    complete_set = pd_normal_data.drop(list(pd_normal_data.columns), axis=1)
    # Insert normalized data
    complete_set[pd_normal_data.columns] = pd.DataFrame(np_normal_data, columns=pd_normal_data.columns)
    # Output data to csv
    complete_set.to_csv(normal_result_path, index=False)


def normal1(x, max_data, min_data, stretch):
    x = (x - min_data) / (max_data - min_data) * stretch
    return x
