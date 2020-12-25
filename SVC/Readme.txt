Data processing
1. Delete irrelevant rows and columns in excel, only keep x and y, put y in the first column, the second column is the
   serial number, the third column starts with x, and the last column is water content
2. Save as to_normalized.csv, set normal in config.ini to True for normalization
3. Delete the normalized csv and save the line with the True value of zero as complete.csv
5. Use the same normalization parameters to normalize the file to be predicted, and save it as to_predict_0.csv

Configuration file
1. Normalization, parameter test and data distribution histogram are separate modules. If the value is True, only this
   module will be run, and the main program will not run
2. When testing parameters, param_grid must contain three parameters: kernel, C, and gamma