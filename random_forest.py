# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:49:07 2018

@author: Gabe Freedman
"""

# Change matplotlib back-end for XMing capability
import matplotlib
# matplotlib.use('TkAgg')

####### GLOBAL IMPORTS #######

# Pickle import based on Python version
import sys
if sys.version_info[0] >= 3:
    import _pickle as cPickle
else:
    import cPickle

# import moby2
# from moby2.instruments import actpol
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
##############################

####### GLOBAL VARIABLES #######
DEPOT = '/data/actpol/depot'
CUTS_TAG = 'mr3c_pa3_f090_s16'
################################


# import train and test TOD lists
train_list = np.loadtxt(r'./data/2016_ar3_train.txt', dtype=str)
test_list = np.loadtxt(r'./data/2016_ar3_test.txt', dtype=str)

# Load pickle file contents
with open(r'./data/mr3_pa2_s16_results.pickle', 'rb') as f:
    data = cPickle.load(f, encoding='latin1')

def generate_test_train_tods(tod_names):
    
    subset = random.sample(tod_names, 100)
    
    random.shuffle(subset)
    
    train_list = subset[:80]
    test_list = subset[80:]
    return train_list, test_list

train_list, test_list = generate_test_train_tods(data['name'])

# Generate list of indices corresponding to train/test TODs
train_ind = [data['name'].index(tod) for tod in train_list]
test_ind = [data['name'].index(tod) for tod in test_list]

pckl_params = ['jumpDark', 'corrLive', 'rmsLive', 'kurtLive', 
               'normDark', 'skewLive', 'DELive', 'jumpLive', 'gainDark', 'corrDark', 'sel']

small_dct = {param: data[param] for param in pckl_params}

def make_dfs(d, tod_list, tod_ind):
    
    df_list = []
    
    for i in range(len(tod_list)):
        td = {k:d[k][:,tod_ind[i]] for k in d.keys()}
        df_list.append(pd.DataFrame(td))
    
    df = pd.concat(df_list)
    return df

train_df = make_dfs(small_dct, train_list, train_ind)
test_df = make_dfs(small_dct, test_list, test_ind)


def split_features_labels(train,test):
    
    x_train = train.drop('sel', axis=1)
    x_test = test.drop('sel', axis=1)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    y_train = np.array(train['sel'])
    y_test = np.array(test['sel'])
    
    return x_train, x_test, y_train, y_test

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