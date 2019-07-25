# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:36:17 2019

@author: Kaai
"""
# import pandas to read the data
import pandas as pd

# import machine learning libraries
from sklearn.preprocessing import StandardScaler, Normalizer

# import functions for vectorizing formula
import composition


mat_props = ['ael_bulk_modulus_vrh'] #'agl_thermal_conductivity_300K','ael_shear_modulus_vrh',

for mat_prop in mat_props:

    # read in the bulk modulus data
    df_t = pd.read_csv('data/'+mat_prop+'/train.csv', usecols=['cif_id', 'target'])
    df_v = pd.read_csv('data/'+mat_prop+'/val.csv', usecols=['cif_id', 'target'])
    

    df = pd.concat([df_t,df_v],ignore_index=True,verify_integrity=True)

    df['cif_id'] = df['cif_id'].str.split('_').str[0]


    # read in compositions from the PCD
    #df_pcd = pd.read_csv('pcd_data/PCD_valid_formulae.csv')
    #df_pcd['target'] = 'unkown'
    #df_pcd['formula'] = df_pcd['formula'].str.replace('[', '(')
    #df_pcd['formula'] = df_pcd['formula'].str.replace(']', ')')


    # sort the bulk values
    df.sort_values(by=['target'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # rename columns for use with vf.generate_features()
    df.columns = ['formula', 'target']
    
    
    
    # get composition-based feature vectors (CBFV)
    X, y, formula = composition.generate_features(df)
#    X_pcd, y_pcd, formula_pcd = composition.generate_features(df_pcd)

    # reset the index because a couple formulae couldn't be processed
    y.reset_index(inplace=True, drop=True)
    X.reset_index(inplace=True, drop=True)
    formula.reset_index(inplace=True, drop=True)

    # remake the full dataframe including features and targets
    df_featurized = X.copy()
    df_featurized['formula'] = formula
    df_featurized['target'] = y

    # sort by the target value so the 'test' set is all extrapolation
    df_featurized.sort_values(by=['target'], inplace=True)

    # reset the index
    df_featurized.reset_index(inplace=True, drop=True)

    # remove the top 100 "extraordinary" compounds from the training data
    df_train = df_featurized.iloc[0:-100, :]

    # set 5000 "ordinary" compounds for the test data
    df_test_false = df_train.sample(500, random_state=1)
    # remove these compounds from the train data
    df_train = df_train[~df_train.index.isin(df_test_false.index.values)]

    # set 100 "extraordinary" compounds for the test data
    df_test_true = df_featurized.iloc[-100:, :]

    # compile the test data "ordinary" + "extraordinary"
    df_test = pd.concat([df_test_false, df_test_true])

    # split the train and test data into features X, and target values y
    X_train = df_train.iloc[:, :-2]
    y_train = df_train.iloc[:, -1]
    formula_train  = df_train.iloc[:, -2]
    X_test = df_test.iloc[:, :-2]
    y_test = df_test.iloc[:, -1]
    formula_test  = df_test.iloc[:, -2]

    # Here we convert the problem from a regression (what is the bulk modulus)
    # to a classification problem (is this compound 'exceptional')
    y_train_label = []
    y_test_label = []

    # In the training set label compounds with values above 245 as "extraordinary"
    for value in y_train:
        if value > 245:
            y_train_label.append(1)  # (1 - extraordinary)
        else:
            y_train_label.append(0)  # (0 - ordinary)

    # In the test set label compounds with values above 300 as "extraordinary" (
    for value in y_test:
        if value > 300:
            y_test_label.append(1)
        else:
            y_test_label.append(0)

    # assign the labels
    y_train_label = pd.Series(y_train_label)
    y_test_label = pd.Series(y_test_label)

    # we now want to process our data.
    # algorithms based on gradient descent and need similar feature scales.

    # scale each column of data to have a mean of 0 and a variance of 1
    scaler = StandardScaler()
    # normalize each row in the data
    normalizer = Normalizer()

    # fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(normalizer.fit_transform(X_train_scaled),
                                  columns=X_train.columns.values)
    # transform the test data based on training data fit
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(normalizer.transform(X_test_scaled),
                                  columns=X_test.columns.values)
    # transform the PCD data based on training data fit
#    X_pcd_scaled = scaler.transform(X_pcd)
#    X_pcd_scaled = pd.DataFrame(normalizer.transform(X_pcd_scaled),
#                                  columns=X_pcd.columns.values)

    # save processed data to CSV files so we do not have to repeat these steps.
    X_train_scaled.to_csv('data/'+mat_prop+'X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('data/'+mat_prop+'X_test_scaled.csv', index=False)

    y_train_label.to_csv('data/'+mat_prop+'y_train_labeled.csv', index=False)
    y_test_label.to_csv('data/'+mat_prop+'y_test_labeled.csv', index=False)

    X_train.to_csv('data/'+mat_prop+'X_train.csv', index=False)
    X_test.to_csv('data/'+mat_prop+'X_test.csv', index=False)

    y_train.to_csv('data/'+mat_prop+'y_train.csv', index=False)
    y_test.to_csv('data/'+mat_prop+'y_test.csv', index=False)

    formula_train.to_csv('data/'+mat_prop+'formula_train.csv', index=False)
    formula_test.to_csv('data/'+mat_prop+'formula_test.csv', index=False)

#    X_pcd_scaled.to_csv('pcd_data/'+mat_prop+'X_pcd_scaled.csv', index=False)
#    X_pcd.to_csv('pcd_data/'+mat_prop+'X_pcd.csv', index=False)
#    formula_pcd.to_csv('pcd_data/'+mat_prop+'formula_pcd.csv', index=False)