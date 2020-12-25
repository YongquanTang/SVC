import configparser

import numpy as np
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.externals import joblib
from sklearn.metrics import f1_score, cohen_kappa_score, roc_curve, auc

from delete_nan import delete_nan


def test_model(testpath, testresultcsvpath, xcol, ycol, svc_model_path, rocpath, negative):
    # ------------------------------------------------Test set----------------------------------------------------------
    # Read test set
    pd_test_data = pd.read_csv(testpath)
    np_test_data = pd_test_data.values  # Convert pandas dataframe to numpy

    # Delete lines with spaces
    test_arr = delete_nan(np_test_data, xcol, ycol)
    X_test = test_arr[0]
    Y_test = test_arr[1]
    np_test_data = test_arr[2]

    # Reload the model
    svc_model = joblib.load(svc_model_path)

    # Test
    y_pre_svc = svc_model.predict(X_test)
    y_pre_svc_pro = svc_model.predict_proba(X_test)

    # Output
    pd_test_data = pd.DataFrame(np_test_data, columns=pd_test_data.columns)
    true_test_data = np_test_data[:, ycol - 1:ycol]
    pd_test_data['Predict_SVC'] = y_pre_svc
    pd_test_data['TF'] = abs(y_pre_svc - true_test_data[:, 0])
    pd_test_data.to_csv(testresultcsvpath, index=False)

    # Calculate the accuracy of the test set
    data = np.array([[0 for i in range(1)] for j in range(len(true_test_data))])
    for i in range(len(true_test_data)):
        if true_test_data[i] == negative:
            data[i][0] = 0
        else:
            data[i][0] = 1
    fpr, tpr, threshold = roc_curve(data, y_pre_svc_pro[:, 1])
    roc_auc = auc(fpr, tpr)

    # Output precision to the console
    print("AUC:" + str(roc_auc))
    print("f1_score:" + str(f1_score(true_test_data, y_pre_svc)))
    print("kappa:" + str(cohen_kappa_score(true_test_data, y_pre_svc)))

    # Draw ROC curve
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    # The false positive rate is the abscissa, the true rate is the ordinate to make the curve
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(rocpath, format='png')


if __name__ == '__main__':
    # -----------------------------------------Get configuration file parameters---------------------------------------
    cp = configparser.ConfigParser()
    cp.read("config.ini", encoding='utf-8-sig')

    path = cp.get("parameters", "path")
    x_col = eval(cp.get("parameters", "x_col"))
    y_col = cp.getint("parameters", "y_col")

    positive = cp.getint("parameters", "positive")
    negative = cp.getint("parameters", "negative")

    test_model(path + 'test.csv', path + 'test_result.csv', x_col, y_col, path + 'svc_model.pkl', path + 'roc.png',
               negative)
