import numpy as np
import pandas as pd
import random_forest_adaboost
import utils
import noising
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# For printing on stdout on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

data = pd.read_csv("data/pima-indians-diabetes.csv", header=None)
Y_index = 0
indexes_to_add_feature_noise = [2, 3, 4, 5, 6, 7, 8]
max_stds = data[indexes_to_add_feature_noise].describe().ix["std"] * 0.1
pPos, pNeg = 0, 0
number_of_tries = 20
avg = 0

Rounds = 400
Iterations  = 50

def TryOuts(indexes_to_add_feature_noise, max_stds, pPos, pNeg):

    Val_error = np.empty(0)

    for p in range(number_of_tries):
        # Add noise on features (before scaling)
        noisy_data = data
        for idx in indexes_to_add_feature_noise:
            std = max_stds.ix[idx]
            if std > 0:
                noisy_data[idx] = noising.addNormalNoise(noisy_data[idx], avg, std)
        # Prepare dataset
        noisy_data["Y"] = np.where(noisy_data[Y_index] == 0, -1, 1)
        noisy_data = noisy_data.drop(Y_index, 1)
        noisy_data[noisy_data.drop("Y", 1).columns.values] = scale(noisy_data[noisy_data.drop("Y", 1).columns.values])
        # Split train / test
        X_train, X_test, y_train, y_test = train_test_split(noisy_data.drop("Y", 1), noisy_data["Y"], test_size=0.33)
        # Add noise on labels (On training set ! otherwise doesn't make sense)
        y_train = noising.switchLabels(y_train, pPos, pNeg)
        # Cross Validate
        #best_Round, best_Iteration, max_mean_minus_std = utils.cross_validate_rf(X_train, y_train, Rounds, Iterations, True)
        best_Round, best_Iteration = Rounds, Iterations
        # Val. Error
        rf = random_forest_adaboost.RandomForestAdaboost(best_Round, best_Iteration)
        rf.fit(X_train, y_train)
        error_val = 1 - rf.score(X_test, y_test)
        train_error = 1 - rf.score(X_train, y_train)
        Val_error = np.append(Val_error, error_val)

        print " Round : {0}, Best Round, {1}, Best T : {2}, train error : {3}, val. error : {4}".format(p + 1, best_Round, best_Iteration, train_error, error_val)

    print("RF Adaboost Avg. val error : {0}, Std. : {1}".format(Val_error.mean(), Val_error.std()))


pPosMax, pNegMax = 0, 0

print "Noise level : 0 % of maximum"
noiseLevel = 0
TryOuts(indexes_to_add_feature_noise, max_stds * noiseLevel, pPosMax * noiseLevel, pNegMax * noiseLevel)

print "Noise level : 50 % of maximum"
noiseLevel = 0.5
TryOuts(indexes_to_add_feature_noise, max_stds * noiseLevel, pPosMax * noiseLevel, pNegMax * noiseLevel)

print "Noise level : 100 % of maximum"
noiseLevel = 1
TryOuts(indexes_to_add_feature_noise, max_stds * noiseLevel, pPosMax * noiseLevel, pNegMax * noiseLevel)
