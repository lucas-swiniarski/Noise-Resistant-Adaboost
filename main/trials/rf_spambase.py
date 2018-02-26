import numpy as np
import pandas as pd
import adaboost
import utils
import noising
import matplotlib.pyplot as plt
import random_forest_adaboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# For printing on stdout on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

data = pd.read_csv("data/spambase.data.shuffled", header=None)
Y_index = 57
avg, std = 0, 0
#pPos, pNeg = 0.1, 0
number_of_tries = 10

Rounds = 1000
Iterations = 100

def TryOuts(pPos, pNeg):
    Val_error = np.empty(0)
    for p in range(number_of_tries):
        # Add noise on features (before scaling)
        noisy_data = data
        # Prepare dataset
        noisy_data["Y"] = np.where(noisy_data[Y_index] == 0, -1, 1)
        noisy_data = noisy_data.drop(Y_index, 1)
        noisy_data[noisy_data.drop("Y", 1).columns.values] = scale(noisy_data[noisy_data.drop("Y", 1).columns.values])
        # Split train / test
        X_train, X_test, y_train, y_test = train_test_split(noisy_data.drop("Y", 1), noisy_data["Y"], test_size=0.33)
        # Add noise on labels (On training set ! otherwise doesn't make sense)
        y_train = noising.switchLabels(y_train, pPos, pNeg)
        # Cross Validate -- Don't compute ..
        #best_Round, best_Iteration, max_mean_minus_std = utils.cross_validate_rf(X_train, y_train, Rounds, Iterations, True)
        best_Round, best_Iteration = Rounds, Iterations
        # Val. Error
        rf = random_forest_adaboost.RandomForestAdaboost(best_Round, best_Iteration)
        rf.fit(X_train, y_train)
        error_val = 1 - rf.score(X_test, y_test)
        Val_error = np.append(Val_error, error_val)
        train_error = 1 - rf.score(X_train, y_train)
        print " Round : {0}, Best Round, {1}, Best T : {2}, train error : {3}, val. error : {4}".format(p + 1, best_Round, best_Iteration, train_error, error_val)

    print("RF Adaboost Avg. val error : {0}, Std. : {1}".format(Val_error.mean(), Val_error.std()))

pPosMax, pNegMax = 0.1, 0.1

print "Noise level : 0 % of maximum"
noiseLevel = 0
TryOuts(pPosMax * noiseLevel, pNegMax * noiseLevel)

print "Noise level : 50 % of maximum"
noiseLevel = 0.5
TryOuts(pPosMax * noiseLevel, pNegMax * noiseLevel)

print "Noise level : 100 % of maximum"
noiseLevel = 1
TryOuts(pPosMax * noiseLevel, pNegMax * noiseLevel)
