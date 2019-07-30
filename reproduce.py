#!/usr/bin/env python
# coding: utf-8



# import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import argparse

# import machine learning libraries
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import f1_score
from matplotlib.colors import Normalize

# import model selection tools
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, learning_curve, GridSearchCV, KFold

# grab metrics to evaluate our models
from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error, precision_recall_fscore_support

# import custom functions for vectorizing & visualizing data
import composition
import utils



all_props = ['bulk modulus','thermal conductivity','shear modulus','band gap','debye temperature','thermal expansion']
arg2prop = {'bulk modulus':'ael_bulk_modulus_vrh', 'thermal conductivity':'agl_log10_thermal_conductivity_300K', 
            'shear modulus':'ael_shear_modulus_vrh', 'band gap':'Egap', 'debye temperature':'ael_log10_debye_temperature', 
            'thermal expansion':'agl_log10_thermal_expansion_300K'}


parser = argparse.ArgumentParser(description = 'Reproduce the results of this work')
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--properties', type=str, nargs='+', metavar = 'Property to reproduce', choices=all_props,
                    help='Select material properties to run and generate figures. Either use --all for every property to run or specify properties, which should be in quotes. For example: python reproduce.py --properties "bulk modulus" "band gap"\r\n \
                    Options: "bulk modulus"\r\n "thermal conductivity"\r\n "shear modulus"\r\n "band gap"\r\n "debye temperature"\r\n "thermal expansion".\r\n')

group.add_argument('--all', action='store_true', help='Run through each property one at a time and generate results and figures.')
                    
args = parser.parse_args()



pcd = False

def optimize_threshold(y_train_labeled, y_train_pred):
    """Given a DataFrame of labels and predictions, return the optimal threshold
    for a high F1 score"""
    y_train_ = y_train_labeled.copy()
    y_train_pred_ = pd.Series(y_train_pred).copy()
    f1score_max = 0
    for threshold in np.arange(0.1, 1, 0.1):
        threshold = min(y_train_pred) + threshold * (max(y_train_pred) - min(y_train_pred))
        y_train_pred_[y_train_pred < threshold] = 0
        y_train_pred_[y_train_pred >= threshold] = 1
        f1score = f1_score(y_train_, y_train_pred_)
        if f1score > f1score_max:
            f1score_max = f1score
            opt_thresh = threshold
    return opt_thresh
    
    
if not args.all: 
    mat_props = []
    for j in args.properties:
        mat_props.append(arg2prop[j])
else:
    mat_props = list(map(lambda p: arg2prop[p],all_props))
print(mat_props)
exit()
for mat_prop in mat_props:
    #Prepare data
    os.makedirs('figures/'+mat_prop, exist_ok=True)
    X_train = pd.read_csv('data/'+mat_prop+'_X_train.csv')
    X_train_scaled = pd.read_csv('data/'+mat_prop+'_X_train_scaled.csv')
    y_train = pd.read_csv('data/'+mat_prop+'_y_train.csv', header=None, squeeze=True)
    y_train_labeled = pd.read_csv('data/'+mat_prop+'_y_train_labeled.csv', header=None, squeeze=True)

    X_test = pd.read_csv('data/'+mat_prop+'_X_test.csv')
    X_test_scaled = pd.read_csv('data/'+mat_prop+'_X_test_scaled.csv')
    y_test = pd.read_csv('data/'+mat_prop+'_y_test.csv', header=None, squeeze=True)
    y_test_labeled = pd.read_csv('data/'+mat_prop+'_y_test_labeled.csv', header=None, squeeze=True)

    formula_train = pd.read_csv('data/'+mat_prop+'_formula_train.csv', header=None, squeeze=True)
    formula_test = pd.read_csv('data/'+mat_prop+'_formula_test.csv', header=None, squeeze=True)

    if pcd is True:
        X_pcd = pd.read_csv('pcd_data/X_pcd.csv')
        X_pcd_scaled = pd.read_csv('pcd_data/X_pcd_scaled.csv')
        formula_pcd = pd.read_csv('pcd_data/formula_pcd.csv', header=None, squeeze=True)

    plt.rcParams.update({'font.size': 12})

    test_threshold = y_test.iloc[-y_test_labeled.sum().astype(int)]
    train_threshold = y_train.iloc[-y_train_labeled.sum().astype(int)]

    y = pd.concat([y_train, y_test])
    plt.figure(1, figsize=(7, 7))
    ax = sns.distplot(y, bins=50, kde=False)

    rect1 = patches.Rectangle((test_threshold, 0), ax.get_xlim()[1]-test_threshold, ax.get_ylim()[1], linewidth=1, edgecolor='k', facecolor='g', alpha=0.2)
    ax.add_patch(rect1)

    text_size = 18

    ax.text(.1, .5, 'Ordinary\nCompounds', size=text_size, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.text(.98, .15, 'Extraordinary\nCompounds', size=text_size, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    ax.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True, labelsize=text_size)
    ax.set_xlabel(mat_prop.title(), size=text_size)
    ax.set_ylabel('number of occurances'.title(), size=text_size)
    plt.savefig('figures/' + mat_prop +'/distplot', dpi=300, bbox_inches='tight')
    plt.show()

    # ## Learn with a Ridge Regression (linear model)

    # define ridge regression object
    rr = Ridge()
    # define k-folds
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    # choose search space
    parameter_candidates = {'alpha': np.logspace(-5, 2, 10)}

    # define the grid search
    grid = GridSearchCV(estimator=rr,
                        param_grid=parameter_candidates,
                        cv=cv)
    # run grid search
    grid.fit(X_train_scaled, y_train)

    # plot grid search to ensure good values)
    plot = utils.plot_1d_grid_search(grid, midpoint=0.75)
    print('best parameters:', grid.best_params_)
    plt.savefig('figures/' + mat_prop +'/rr_1d_search', dpi=300, bbox_inches='tight')
    plt.show()
    best_params_rr = grid.best_params_


#    best_params_rr = {'alpha': 0.0021544346900318843}
    rr = Ridge(**best_params_rr)
    rr.fit(X_train_scaled, y_train)
    y_test_predicted_rr = rr.predict(X_test_scaled)
    y_train_predicted_rr = rr.predict(X_train_scaled)
    # plot the data
    plt.figure(figsize=(6,6))
    plt.plot(y_test, y_test_predicted_rr, marker='o', mfc='none', color='#0077be', linestyle='none', label='test')
    plt.plot(y_train, y_train_predicted_rr, marker='o', mfc='none', color='#e34234', linestyle='none', label='train')
    max_val = max(y_test.max(), y_test_predicted_rr.max())
    min_val = min(y_test.min(), y_test_predicted_rr.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    limits = [min_val, max_val]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.legend(loc=4)
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.savefig('figures/' + mat_prop +'/rr_act_vs_pred', dpi=300, bbox_inches='tight')
    plt.show()

    # ## Learn with a support vector regression (non-linear model)



    # to speed up the grid search, optimize on a subsample of data
    X_train_scaled_sampled = X_train_scaled.sample(500, random_state=1)
    y_train_sampled = y_train.loc[X_train_scaled_sampled.index.values]

    # define support vector regression object (default to rbf kernel)
    svr = SVR()
    # define k-folds
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    # choose search space
    parameter_candidates = {'C': np.logspace(2, 4, 8), 'gamma': np.logspace(-3, 1, 8)}

    # define the grid search
    grid = GridSearchCV(estimator=svr,
                        param_grid=parameter_candidates,
                        cv=cv)
    # run grid search
    grid.fit(X_train_scaled_sampled, y_train_sampled)

    # plot grid search to ensure good values
    utils.plot_2d_grid_search(grid, midpoint=0.7)
    plt.savefig('figures/' + mat_prop +'/svr_2d_search', dpi=300, bbox_inches='tight')
    plt.show()
    print('best parameters:', grid.best_params_)
    best_params_svr = grid.best_params_


    svr = SVR(**best_params_svr)
    svr.fit(X_train_scaled, y_train)

    y_test_predicted_svr = svr.predict(X_test_scaled)
    y_train_predicted_svr = svr.predict(X_train_scaled)

    # plot the data
    plt.figure(figsize=(6,6))
    plt.plot(y_test, y_test_predicted_svr, marker='o', mfc='none', color='#0077be', linestyle='none', label='test')
    plt.plot(y_train, y_train_predicted_svr, marker='o', mfc='none', color='#e34234', linestyle='none', label='train')

    max_val = max(y_test.max(), y_test_predicted_svr.max())
    min_val = min(y_test.min(), y_test_predicted_svr.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    limits = [min_val, max_val]
    plt.xlim(limits)
    plt.ylim(limits)

    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.legend(loc=4)
    plt.tick_params(direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.savefig('figures/' + mat_prop +'/svr_act_vs_pred', dpi=300, bbox_inches='tight')
    plt.show()


    # # Approach the problem as a classification task

    # ## Learn with a logistic regression (linear classification) 



    # define logistic regression object
    lr = LogisticRegression(solver='lbfgs')
    # define k-folds
    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # choose search space
    class_weights = []
    class_1_weight = [{0:1, 1:weight} for weight in np.linspace(1, 50, 5)]
    parameter_candidates = {'C': np.logspace(-1, 4, 5), 
                            'class_weight': class_1_weight}

    # define the grid search. We use log-loss to decide which parameters to use.
    grid = GridSearchCV(estimator=lr,
                        param_grid=parameter_candidates,
                        scoring='neg_log_loss',
                        cv=cv)

    # run grid search
    grid.fit(X_train_scaled, y_train_labeled)

    # plot grid search to ensure good values
    utils.plot_2d_grid_search(grid, midpoint=-0.05, vmin=-0.13, vmax=0)
    plt.savefig('figures/' + mat_prop +'/lr_2d_search', dpi=300, bbox_inches='tight')
    plt.show()
    print('best parameters:', grid.best_params_)
    best_params_lr = grid.best_params_




    lr = LogisticRegression(solver='lbfgs', penalty='l2', **best_params_lr)
    lr.fit(X_train_scaled, y_train_labeled)

    # define k-folds
    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    y_pred_train_lr = cross_val_predict(lr, X_train_scaled, y_train_labeled, cv=cv)
    y_prob_train_lr = cross_val_predict(lr, X_train_scaled, y_train_labeled, cv=cv, method='predict_proba')
    y_probability_train_lr = [probability[1] for probability in y_prob_train_lr]

    y_pred_test_lr = lr.predict(X_test_scaled)
    y_prob_test_lr = lr.predict_proba(X_test_scaled)
    y_probability_test_lr = [probability[1] for probability in y_prob_test_lr]

    df_cm = pd.DataFrame(confusion_matrix(y_train_labeled, y_pred_train_lr))

    ax = sns.heatmap(df_cm, square=True, annot=True, annot_kws={"size": 18}, cbar=False, linewidths=.5, cmap="YlGnBu", center=-10000000)

    ax.set_ylabel('actual')
    ax.set_xlabel('predicted')
    ax.xaxis.tick_top()
    plt.savefig('figures/' + mat_prop +'/lr_cm', dpi=300, bbox_inches='tight')
    plt.show()


    threshold = 0.5
    utils.plot_prob(threshold, y_train, y_probability_train_lr, threshold_x=train_threshold, mat_prop=mat_prop)
    plt.title('Logistic Regression: training set')
    plt.savefig('figures/' + mat_prop +'/lr_train_prob_thresh={:0.2f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()

    # ### Check our perfromance on the test set!


    utils.plot_prob(threshold, y_test, y_probability_test_lr, threshold_x=test_threshold, mat_prop=mat_prop)
    plt.title('Logistic Regression: test set')
    plt.savefig('figures/' + mat_prop +'/lr_test_prob_thresh={:0.2f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()


    # ### Compare this performance to regression models
    # 
    # **For the same recall, we are three times more likely that predicted compound is not actually extraordinary.**


    threshold = optimize_threshold(y_train_labeled, y_train_predicted_rr)
    utils.plot_regression(threshold, y_train, y_train_predicted_rr, threshold_x=train_threshold, mat_prop=mat_prop)
    plt.title('Ridge Regression: training set')
    plt.savefig('figures/' + mat_prop +'/rr_train_reg_thresh={:0.2f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()


    utils.plot_regression(threshold, y_test, y_test_predicted_rr, threshold_x=test_threshold, mat_prop=mat_prop)
    plt.title('Ridge Regression: test set')
    plt.savefig('figures/' + mat_prop +'/rr_test_reg_thresh={:0.2f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()



    threshold = optimize_threshold(y_train_labeled, y_train_predicted_svr)
    utils.plot_regression(threshold, y_train, y_train_predicted_svr, threshold_x=train_threshold, mat_prop=mat_prop)
    plt.title('SVR: training set')
    plt.savefig('figures/' + mat_prop +'/svr_train_reg_thresh={:0.02f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()


    utils.plot_regression(threshold, y_test, y_test_predicted_svr, threshold_x=test_threshold, mat_prop=mat_prop)
    plt.title('SVR: test set')
    plt.savefig('figures/' + mat_prop +'/svr_test_reg_thresh={:0.02f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()


    # ## Learn with a support vector classification (non-linear)


    # to speed up the grid search, optimize on a subsample of data 
    y_train_labeled_sampled = y_train_labeled.loc[X_train_scaled_sampled.index.values]

    # define suppor vector classification object (need to set probability to True)
    svc = SVC(probability=True)
    # define k-folds
    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # choose search space (we will start with class_weight=1 
    # as that was optimal for svc)
    parameter_candidates = {'C': np.logspace(-1, 4, 5), 
                            'gamma': np.logspace(-2, 2, 5)}

    # define the grid search. We use log-loss to decide which parameters to use.
    grid = GridSearchCV(estimator=svc,
                        param_grid=parameter_candidates,
                        scoring='neg_log_loss',
                        cv=cv)

    # run grid search
    grid.fit(X_train_scaled_sampled, y_train_labeled_sampled)

    # plot grid search to ensure good values
    utils.plot_2d_grid_search(grid, midpoint=-0.04, vmin=-0.13, vmax=0)
    plt.savefig('figures/' + mat_prop +'/svc_2d_search.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('best parameters:', grid.best_params_)
    best_params_svc = grid.best_params_


    svc = SVC(probability=True, **best_params_svc)
    svc.fit(X_train_scaled, y_train_labeled)

    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    y_pred_train_svc = cross_val_predict(svc, X_train_scaled, y_train_labeled, cv=cv)
    y_prob_train_svc = cross_val_predict(svc, X_train_scaled, y_train_labeled, cv=cv, method='predict_proba')
    y_probability_train_svc = [probability[1] for probability in y_prob_train_svc]

    y_pred_test_svc = svc.predict(X_test_scaled)
    y_prob_test_svc = svc.predict_proba(X_test_scaled)
    y_probability_test_svc = [probability[1] for probability in y_prob_test_svc]

    precision, recall, fscore, support = precision_recall_fscore_support(y_train_labeled, y_pred_train_svc)
    print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1], recall[1]))
    df_cm = pd.DataFrame(confusion_matrix(y_train_labeled, y_pred_train_svc))

    ax = sns.heatmap(df_cm, square=True, annot=True, annot_kws={"size": 18}, cbar=False, linewidths=.5, cmap="YlGnBu", center=-10000000)
    ax.set_ylabel('actual')
    ax.set_xlabel('predicted')
    ax.xaxis.tick_top()
    plt.savefig('figures/' + mat_prop +'/svc_cm', dpi=300, bbox_inches='tight')
    plt.show()



    threshold = 0.5
    utils.plot_prob(threshold, y_train, y_probability_train_svc, threshold_x=train_threshold, mat_prop=mat_prop)
    plt.title('SVC: training set')
    plt.savefig('figures/' + mat_prop +'/svc_train_prob_thresh={:0.02f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()




    utils.plot_prob(threshold, y_test, y_probability_test_svc, threshold_x=test_threshold, mat_prop=mat_prop)
    plt.title('SVC: test set')
    plt.savefig('figures/' + mat_prop +'/svc_test_prob_thresh={:0.2f}.png'.format(threshold), dpi=300, bbox_inches='tight')
    plt.show()


    # # 6. Bonus Material: generating predictions from the PCD
    # 
    # ### You will need the PCD data (available by email). You will also need to set pcd=True in this file cell.


    if pcd is True:
        X_all_scaled = pd.concat([X_train_scaled, X_test_scaled], ignore_index=True)
        y_all = pd.concat([y_train, y_test], ignore_index=True)
        formula_all = pd.concat([formula_train, formula_test], ignore_index=True)

        y_all_labeled = []
        for value in y_all:
            if value > 275:
                y_all_labeled.append(1)
            else:
                y_all_labeled.append(0)
        y_all_labeled = pd.Series(y_all_labeled)
        print(y_all_labeled.value_counts())

        best_params_svc = {'C': 562.341325190349, 'gamma': 0.1}
        svc_all = SVC(probability=True, **best_params_svc, random_state=1)
        svc_all.fit(X_all_scaled, y_all_labeled)

        best_params_lr = {'C': 31.622776601683793, 'class_weight': {0: 1, 1: 1.0}}
        lr_all = LogisticRegression(**best_params_lr, random_state=1)
        lr_all.fit(X_all_scaled, y_all_labeled)

        y_prob_pcd_svc = svc_all.predict_proba(X_pcd_scaled)
        y_prob_pcd_lr = lr_all.predict_proba(X_pcd_scaled)

        extraordinary_formulae_svc = []
        extraordinary_formulae_lr = []
        threshold = 0.25
        for prob, formula in zip(y_prob_pcd_svc, formula_pcd):
            if prob[1] >= threshold:
                extraordinary_formulae_svc.append(formula)

        for prob, formula in zip(y_prob_pcd_lr, formula_pcd):
            if prob[1] >= threshold:
                extraordinary_formulae_lr.append(formula)

        print('\n'+'# extraordinary compounds scv={:}'.format(len(extraordinary_formulae_svc)))
        print('# extraordinary compounds lr={:}'.format(len(extraordinary_formulae_lr)))

        def get_non_DFT_compounds(extraordinary_formulae):
            in_sample = []
            out_of_sample = []
            fractional_formula_all  = [composition._fractional_composition(formula) for formula in formula_all]
            for formula in extraordinary_formulae:
                fractional_formula = composition._fractional_composition(formula)
                if fractional_formula not in fractional_formula_all:
                    out_of_sample.append(formula)
                else:
                    in_sample.append(fractional_formula)
                    in_sample.append(formula)
            return out_of_sample

        out_svc = get_non_DFT_compounds(extraordinary_formulae_svc)
        out_lr = get_non_DFT_compounds(extraordinary_formulae_lr)

        overlap = []
        for lr in out_lr:
            if lr in out_svc:
                overlap.append(lr)
        print('# of overlapping compounds:', len(overlap))

        print('\ncompounds of interest:', overlap)
    
    
    
    
