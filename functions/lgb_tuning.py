## self parameter tuning function for lgb model with hyperopt and lightgbm

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import csv
import hyperopt as hp
from hyperopt import tpe
from hyperopt import Trials
from timeit import default_timer as timer
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import fmin
from sklearn.model_selection import GroupKFold





def lgb_tuning(lgb_cv,N_FOLDS=5,MAX_EVALS=100,output_file='bayes_test.csv',metric='auc',objection='binary',groups=None):
    def objective(hyperparameters,groups=groups):
        # Keep track of evals
        ITERATION =0

        # Using early stopping to find number of trees trained
        if 'n_estimators' in hyperparameters:
            del hyperparameters['n_estimators']

        # Retrieve the subsample
        subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

        # Extract the boosting type and subsample to top level keys
        hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
        hyperparameters['subsample'] = subsample

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples','max_depth']:
            hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        hyperparameters['objective']=objection
        #hyperparameters['verbose']=-1
        start = timer()
        
        # Perform n_folds cross validation
        if groups:           
            groups=lgb_cv.get_group()
            folds=GroupKFold().split(lgb_cv.get_label(),groups=groups)
        else:
            folds=None
        
        if metric.lower()=='map':
            hyperparameters['eval_at']=1
        
        
        cv_results = lgb.cv(hyperparameters, lgb_cv, num_boost_round = 4000, nfold = N_FOLDS,folds=folds,\
                            early_stopping_rounds=300, metrics = metric)

        run_time = timer() - start
        
        score_key=sorted(cv_results.keys())[0]
        # Extract the best score
        best_score = cv_results[score_key][-1]

        # Loss must be minimized
        if metric=='binary_error':
            loss=best_score
        else:
            loss = 1 - best_score

        # Boosting rounds that returned the highest cv score
        n_estimators = len(cv_results[score_key])

        # Add the number of estimators to the hyperparameters
        hyperparameters['n_estimators'] = n_estimators

        # Write to the csv file ('a' means append)
        of_connection = open(OUT_FILE, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
        of_connection.close()

        # Dictionary with information for evaluation
        return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
                'train_time': run_time, 'status': STATUS_OK}
    
    # Define the search space
    space = {
        'boosting_type': hp.choice('boosting_type', 
                                    [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                    {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                         {'boosting_type': 'goss', 'subsample': 1.0}]),
        'num_leaves': hp.quniform('num_leaves', 20, 200, 4),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.5)),
        #'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 300, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 0.2),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 0.2),
        #'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        'is_unbalance': hp.choice('is_unbalance', [True, False]),
        'max_depth': hp.quniform('max_depth', 4, 8, 1)
        }
       

    # Create the algorithm
    tpe_algorithm = tpe.suggest
    # Record results
    trials = Trials()
    
    
    # Create a file and open a connection
    OUT_FILE = output_file
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)


    # Write column names
    headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
    writer.writerow(headers)
    of_connection.close()
    #global  ITERATION

    ITERATION = 0
    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
                max_evals = MAX_EVALS)
    return best