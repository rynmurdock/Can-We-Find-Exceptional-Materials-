# import pandas to read the data
import pandas as pd
# import machine learning libraries
from sklearn.preprocessing import StandardScaler, Normalizer
# import functions for vectorizing formula
import composition

mat_props = ['ael_bulk_modulus_vrh', 'agl_log10_thermal_conductivity_300K', 'ael_shear_modulus_vrh', 'Egap', 'ael_log10_debye_temperature', 'agl_log10_thermal_expansion_300K']  # 'ael_bulk_modulus_vrh', 'agl_log10_thermal_conductivity_300K', 'ael_shear_modulus_vrh', 'Egap', 'ael_log10_debye_temperature', 'agl_log10_thermal_expansion_300K'

for mat_prop in mat_props:

    # read in the property data
    df_t = pd.read_csv('data/'+mat_prop+'/train.csv', usecols=['cif_id', 'target'])
    df_v = pd.read_csv('data/'+mat_prop+'/val.csv', usecols=['cif_id', 'target'])

    if mat_prop == 'Egap':
        df_t = df_t[df_t['target'] > 0]
        df_t = df_t[df_t['target'] < 10]
        df_v = df_v[df_v['target'] > 0]

    df = pd.concat([df_t,df_v],ignore_index=True,verify_integrity=True)

    df['cif_id'] = df['cif_id'].str.split('_').str[0]

    # sort the bulk values
    df.sort_values(by=['target'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # rename columns for use with vf.generate_features()
    df.columns = ['formula', 'target']

    # get composition-based feature vectors (CBFV)
    X, y, formula = composition.generate_features(df)

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

    # remove the top 1% "extraordinary" compounds from the training data
    n_extraordinary = int(df_featurized.shape[0] * 0.01)
    df_train = df_featurized.iloc[0:-n_extraordinary, :]

    # set 15% of the train as "ordinary" compounds for the test data
    n_test = int(df_train.shape[0] * 0.15)
    df_test_false = df_train.sample(n_test, random_state=1)
    # remove these compounds from the train data
    df_train = df_train[~df_train.index.isin(df_test_false.index.values)]

    # set the top 1% "extraordinary" compounds for the test data
    df_test_true = df_featurized.iloc[-n_extraordinary:, :]

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
    y_train_label = y_train.copy()
    y_test_label = y_test.copy()

    # label extraordinary compounds in train and test set
    n_test_extraordinary = df_test_true.shape[0]
    n_train_extraordinary = int(df_test_true.shape[0] / df_test_false.shape[0] * df_train.shape[0])
    y_train_label.iloc[:-n_train_extraordinary] = [0] * (y_train.shape[0] - n_train_extraordinary)
    y_train_label.iloc[-n_train_extraordinary:] = [1] * n_train_extraordinary
    y_test_label.iloc[:-n_test_extraordinary] = [0] * (y_test.shape[0] - n_test_extraordinary)
    y_test_label.iloc[-n_test_extraordinary:] = [1] * n_test_extraordinary

# =============================================================================
#     # we now want to process our data.
#     # algorithms based on gradient descent and need similar feature scales.
# =============================================================================
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

    # save processed data to CSV files so we do not have to repeat these steps.
    X_train_scaled.to_csv('data/'+mat_prop+'_X_train_scaled.csv', index=False)
    X_test_scaled.to_csv('data/'+mat_prop+'_X_test_scaled.csv', index=False)

    y_train_label.to_csv('data/'+mat_prop+'_y_train_labeled.csv', index=False)
    y_test_label.to_csv('data/'+mat_prop+'_y_test_labeled.csv', index=False)

    X_train.to_csv('data/'+mat_prop+'_X_train.csv', index=False)
    X_test.to_csv('data/'+mat_prop+'_X_test.csv', index=False)

    y_train.to_csv('data/'+mat_prop+'_y_train.csv', index=False)
    y_test.to_csv('data/'+mat_prop+'_y_test.csv', index=False)

    formula_train.to_csv('data/'+mat_prop+'_formula_train.csv', index=False)
    formula_test.to_csv('data/'+mat_prop+'_formula_test.csv', index=False)
