# -*- coding: utf-8 -*-


# Change matplotlib back-end for XMing capability
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Pickle import based on Python version
import sys
if sys.version_info[0] >= 3:
    import _pickle as cPickle
else:
    import cPickle

import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# import moby2
# from moby2.instruments import actpol

####### GLOBAL VARIABLES #######
DEPOT = '/data/actpol/depot'
CUTS_TAG = 'mr3c_pa3_f090_s16'
################################


def generate_test_train_tods(tod_names):
    
    subset = random.sample(tod_names, 100)
    
    random.shuffle(subset)
    
    train_list = subset[:80]
    test_list = subset[80:]
    return train_list, test_list

def make_dfs(d, tod_list, tod_ind):
    
    df_list = []
    
    for i in range(len(tod_list)):
        td = {k:d[k][:,tod_ind[i]] for k in d.keys()}
        df_list.append(pd.DataFrame(td))
    
    df = pd.concat(df_list)
    return df

def split_features_labels(train,test):
    
    x_train = train.drop('sel', axis=1)
    x_test = test.drop('sel', axis=1)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    y_train = np.array(train['sel'])
    y_test = np.array(test['sel'])
    
    return x_train, x_test, y_train, y_test

def random_forest(tod_train_lst, tod_test_lst, pckl_file):
    
    # Load pickle file contents
    with open(pckl_file, 'rb') as f:
        data = cPickle.load(f, encoding='latin1')
    
    if not tod_train_lst or not tod_test_lst:
        tod_train_lst, tod_test_lst = generate_test_train_tods(data['name'])
 
    # Generate list of indices corresponding to train/test TODs
    train_ind = [data['name'].index(tod) for tod in tod_train_lst]
    test_ind = [data['name'].index(tod) for tod in tod_test_lst]

    # Features for model come from pickle file
    pckl_params = ['corrLive',  'rmsLive',  'kurtLive',  'DELive', 'MFELive', 
                   'skewLive', 'normLive', 'darkRatioLive',   'jumpLive', 
                   'gainLive', 'sel']

    # Extract selected feature arrays from pickle file dictionary
    small_dct = {param: data[param] for param in pckl_params}
    
    train_df = make_dfs(small_dct, tod_train_lst, train_ind)
    test_df = make_dfs(small_dct, tod_test_lst, test_ind)
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    for col in train_df:
        if train_df[col].dtype=='float32':
            train_df[col] = train_df[col].astype('float64')
    for col in test_df:
        if test_df[col].dtype=='float32':
            test_df[col] = test_df[col].astype('float64')
    
    x_train, x_test, y_train, y_test = split_features_labels(train_df, test_df)
    
    regressor = RandomForestClassifier(n_estimators=50, random_state=0)  
    regressor.fit(x_train, y_train)  
    y_pred = regressor.predict(x_test)

    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(len(y_pred)):
        if y_test[i]==y_pred[i]==1:
               tp += 1
        if y_pred[i]==1 and y_test[i]!=y_pred[i]:
               fp += 1
        if y_test[i]==y_pred[i]==0:
               tn += 1
        if y_pred[i]==0 and y_test[i]!=y_pred[i]:
               fn += 1

    total = tp+fp+fn+tn
    print('True Positive: ' +str(tp) + ', ' + str(tp/total))
    print('False Positive: ' +str(fp) + ', ' + str(fp/total))
    print('False Negative: ' +str(fn) + ', ' + str(fn/total))
    print('True Negative: ' +str(tn) + ', ' + str(tn/total))
    print(classification_report(y_test,y_pred))  
    print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    
    # random_forest.py run from command line with file path arguments
    if len(sys.argv) == 4:
        # Set parameters from sys.argv for random forest input
        train_lst = np.loadtxt(sys.argv[1], dtype=str)
        test_lst = np.loadtxt(sys.argv[2], dtype=str)
        pckl_file = sys.argv[3]
    # random_forest.py run from commandline w/o additional arguments
    elif len(sys.argv) == 1:
        # Import existing data files
        train_lst = None
        test_lst = None
        pckl_file = '../data/mr3_pa2_s16_results.pickle'
    else:
        raise ValueError('3 arguments required for random_forest.py: ' + 
                         str(len(sys.argv)-1) + ' provided.')
    
    # Run random forest model with given inputs
    random_forest(train_lst, test_lst, pckl_file)
