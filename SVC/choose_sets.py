import numpy as np
import pandas as pd


# Partition data set
def choose_sets(filepath, trainpath, testpath, testproporation):
    # Input a csv, divided into training set, validation set and test set according to a certain proportion
    print('Start dividing the data set...')

    # Read the csv
    complete_set = pd.read_csv(filepath)

    # Get the index
    ncomplete = len(complete_set.index)

    # Get index randomly
    test_rate = testproporation  # Test set ratio
    ntest = int(ncomplete * test_rate)

    train_index = np.arange(ncomplete)
    # Random and non-repeated extraction, get test set index
    test_index = np.random.choice(train_index, ntest, replace=False)
    # Remove the test set index from the current training set index
    train_index = np.setdiff1d(train_index, test_index)

    # Create a new training set, validation set and test set DataFrame
    train_set = pd.DataFrame(columns=complete_set.columns)
    test_set = pd.DataFrame(columns=complete_set.columns)

    # Copy data
    for ii in train_index:
        train_set = train_set.append(complete_set.iloc[ii], ignore_index=True)
    for ii in test_index:
        test_set = test_set.append(complete_set.iloc[ii], ignore_index=True)

    # Output data to csv
    train_set.to_csv(trainpath, index=False)  # index = False Do not keep indexes
    test_set.to_csv(testpath, index=False)

    print('The data set division is complete!')
