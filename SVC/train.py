import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# from Plot import plot
from delete_nan import delete_nan


# Training a neural network model
def train(filepath, xcol, ycol, param_grid, fw, n_C, n_gamma, w_balanced, w_positive, w_negative, positive, negative,
          score_path, variance_path, n_splits, path):
    # ------------------------------------------------Read training set-------------------------------------------------
    global svc_model
    pd_data = pd.read_csv(filepath)
    np_data = pd_data.values  # Convert pandas dataframe to numpy
    np.random.shuffle(np_data)  # Randomly scramble data
    np.random.shuffle(np_data)  # Randomly scramble data
    np.random.shuffle(np_data)  # Randomly scramble data

    # Delete lines with spaces
    arr = delete_nan(np_data, xcol, ycol)
    X_data = arr[0]
    Y_data = arr[1]

    # -------------------------------------------Automatic parameter optimization--------------------------------------
    print("Start parameter automatic optimization...")

    # Cross validation, divide the training/test data set into 10 mutually exclusive subsets
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    # Sample balance
    if w_balanced:
        class_weight = 'balanced'
    else:
        class_weight = {positive: w_positive, negative: w_negative}

    # Used to plot variance and score images
    auc_variance_total = []
    auc_score_total = []

    # Traverse C and gamma in the parameter grid
    for i in param_grid.get('C'):
        for j in param_grid.get('gamma'):
            # Initialization accuracy
            kappa = 0
            f1 = 0
            oa = 0
            roc_auc = 0

            # Define the SVC model
            svc_model1 = SVC(C=i, gamma=j, probability=True, class_weight=class_weight)

            # Used to calculate the variance of AUC in cross validation
            auc_array = []

            # Used to pick the model with the highest score
            max_auc = 0

            # Start to cross validation
            for train_index, test_index in kfold.split(X_data, Y_data):
                train_X, train_Y = X_data[train_index], Y_data[train_index]
                test_X, test_Y = X_data[test_index], Y_data[test_index]
                # Training model
                svc_model1.fit(train_X, train_Y.ravel())

                # Test model
                predict_test = svc_model1.predict(test_X)
                predict_pro_test = svc_model1.predict_proba(test_X)

                # Calculate the test set accuracy
                oa1 = accuracy_score(test_Y, predict_test, normalize=True)
                kappa1 = cohen_kappa_score(test_Y, predict_test)
                f11 = f1_score(test_Y, predict_test)

                # AUC
                data = np.array([[0 for i in range(1)] for j in range(len(test_X))])
                for m in range(len(test_X)):
                    if test_Y[m] == negative:
                        data[m][0] = 0
                    else:
                        data[m][0] = 1

                # Calculate the true rate and false positive rate
                fpr, tpr, threshold = roc_curve(data, predict_pro_test[:, 1])
                roc_auc1 = auc(fpr, tpr)

                # Add the value of auc to the array to calculate the variance of cross-validation auc
                auc_array.append(roc_auc1)

                # Used to calculate the average accuracy of cross validation
                oa += oa1
                kappa += kappa1
                f1 += f11
                roc_auc += roc_auc1

                # Pick the model with the highest AUC score
                if max_auc < roc_auc1:
                    svc_model = svc_model1

            # Calculate the average accuracy of cross-validation
            oa = oa / n_splits
            kappa = kappa / n_splits
            f1 = f1 / n_splits
            roc_auc = roc_auc / n_splits

            # Save variance and score to the list for drawing
            variance = np.std(auc_array) * np.std(auc_array)
            auc_variance_total.append(variance)
            auc_score_total.append(roc_auc)

            # Save the model
            svc_model_path = path + 'model_oa_' + str('%.4f' % oa) + '_kappa_' + str('%.4f' % kappa) + '_f1_' + str(
                '%.4f' % f1) + '_auc_' + str('%.4f' % roc_auc) + '_variance_' + str(variance) + '_c_' + str(
                i) + '_gamma_' + str(j) + '.pkl'
            joblib.dump(svc_model, svc_model_path)

            # Print model parameters to the console
            print('oa:{:.4f},kappa:{:.4f},f1:{:.4f},auc:{:.4f}'
                  .format(oa, kappa, f1, roc_auc) + ',variance:' + str(variance) + ',c:' + str(
                i) + ',gamma:' + str(j))

    # Output score variance to txt file
    fw.write("score：" + str(auc_score_total) + '\n')
    fw.write("variance：" + str(auc_variance_total) + '\n')

    # Output score and variance as grayscale image,Assuming you have installed gdal
    # plot(auc_score_total, auc_variance_total, n_C, n_gamma, score_path, variance_path)
