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
from matplotlib import pyplot as plt
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
