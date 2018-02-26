import numpy as np
import pandas as pd
import adaboost
import utils
import noising
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

data = pd.read_csv("data/bupa.csv", header=None)
Y_index = 0
indexes_to_add_feature_noise = [2, 3, 4, 5, 6]
max_stds = data[indexes_to_add_feature_noise].describe().ix["std"] * 0.1
pPos, pNeg = 0, 0
number_of_tries = 10
avg = 0

Ts = [10 * i for i in range(1, 21)]
Cs = [10 ** i for i in range(-2, 3)]

def TryOuts(indexes_to_add_feature_noise, max_stds, pPos, pNeg):
    reload(noising)
    reload(utils)

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
        best_T, cross_validate_accuracy_training = utils.cross_validate_adaboost_T(X_train, y_train, Ts, None, None, 0, False)
        # Val. Error
        my_adaboost = adaboost.Adaboost(best_T)
        my_adaboost.fit(X_train, y_train)
        error_val = 1 - my_adaboost.score(X_test, y_test)
        Val_error = np.append(Val_error, error_val)
        #print " Round : {0}, Best T : {1}, train error : {2}, val. error : {3}".format(p + 1, best_T, 1 - cross_validate_accuracy_training, error_val)

    print("Adaboost Avg. val error : {0}, Std. : {1}".format(Val_error.mean(), Val_error.std()))

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
        best_C, best_T, cross_validate_accuracy_training = utils.cross_validate_adaboost_T_C(X_train, y_train, Ts, 2, Cs, 0, False)
        # Val. Error
        my_adaboost = adaboost.Adaboost(best_T, 2, best_C, 0)
        my_adaboost.fit(X_train, y_train)
        error_val = 1 - my_adaboost.score(X_test, y_test)
        Val_error = np.append(Val_error, error_val)
        #print " Round : {0}, Best T / C : {1}, {2}, train error : {3}, val. error : {4}".format(p + 1, best_T, best_C, 1 - cross_validate_accuracy_training, error_val)

    print("L1 regulariazed Adaboost (paper) Avg. val error : {0}, Std. : {1}".format(Val_error.mean(), Val_error.std()))

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
        best_C, best_T, cross_validate_accuracy_training = utils.cross_validate_adaboost_T_C(X_train, y_train, Ts, 2, Cs, 1, False)
        # Val. Error
        my_adaboost = adaboost.Adaboost(best_T, 2, best_C, 1)
        my_adaboost.fit(X_train, y_train)
        error_val = 1 - my_adaboost.score(X_test, y_test)
        Val_error = np.append(Val_error, error_val)
        #print " Round : {0}, Best T : {1}, train error : {2}, val. error : {3}".format(p + 1, best_T, 1 - cross_validate_accuracy_training, error_val)

    print("Adaboost v1 Avg. val error : {0}, Std. : {1}".format(Val_error.mean(), Val_error.std()))

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
        best_C, best_T, cross_validate_accuracy_training = utils.cross_validate_adaboost_T_C(X_train, y_train, Ts, 3, Cs, 1, False)
        # Val. Error
        my_adaboost = adaboost.Adaboost(best_T, 3, best_C, 1)
        my_adaboost.fit(X_train, y_train)
        error_val = 1 - my_adaboost.score(X_test, y_test)
        Val_error = np.append(Val_error, error_val)
        #print " Round : {0}, Best T : {1}, train error : {2}, val. error : {3}".format(p + 1, best_T, 1 - cross_validate_accuracy_training, error_val)

    print("Adaboost v1 pow 3Avg. val error : {0}, Std. : {1}".format(Val_error.mean(), Val_error.std()))

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
