import configparser

import numpy as np
import pandas as pd
from sklearn.externals import joblib


# The predict results are saved to a new csv
def predict(topredictpath, xcol, ycol, predictresultpath, svc_model_path, negative, positive):
    # Read the csv to be predicted
    pd_predict_data = pd.read_csv(topredictpath)
    np_predict_data = pd_predict_data.values  # Convert pandas dataframe to numpy

    # Reload the model
    svc_model = joblib.load(svc_model_path)

    # nan value is set to 0
    nanlist = np.argwhere(pd.isna(np_predict_data))
    for ii in nanlist:
        np_predict_data[ii[0]][ii[1]] = 0
    X = np_predict_data[:, xcol]

    # predict
    predict_value = svc_model.predict(X)  # Output the predicted value of SVC
    predict_proba = svc_model.predict_proba(X)  # Output the probability value of SVC

    # Output
    pd_predict_data[pd_predict_data.columns[ycol - 1]] = 0
    pd_predict_data['Predict_SVC'] = predict_value
    pd_predict_data['pro_SVC'] = predict_proba[:, 1]  # SVC

    # Create a new array the same as predict_proba, and output the prediction result with the probability as the
    # boundary of 0.5
    predict_proba_1 = np.array([[0 for i in range(1)] for j in range(len(predict_proba))])
    for i in range(len(predict_proba)):
        if predict_proba[i][1] < 0.5:
            predict_proba_1[i][0] = negative
        else:
            predict_proba_1[i][0] = positive

    pd_predict_data['predict_pro_SVC'] = predict_proba_1

    # Output to csv
    pd_predict_data.to_csv(predictresultpath, index=False)


if __name__ == '__main__':
    # -----------------------------------------Get configuration file parameters---------------------------------------
    cp = configparser.ConfigParser()
    cp.read("config.ini", encoding='utf-8-sig')

    path = cp.get("parameters", "path")
    x_col = eval(cp.get("parameters", "x_col"))
    y_col = cp.getint("parameters", "y_col")

    positive = cp.getint("parameters", "positive")
    negative = cp.getint("parameters", "negative")

    predict(path + 'to_predict_0.csv', x_col, y_col, path + 'predict_result_0.csv',
            path + 'svc_model.pkl', negative, positive)
