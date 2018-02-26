import numpy as np
import pandas as pd
# Allow matplotlib to plot on HPC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import adaboost
import random_forest_adaboost
from sklearn.cross_validation import KFold

def plot_cross_validation_curve(fold_results_train, params, label, log_scale, save):
    plt.close("all")
    train_accuracy_plus = []
    train_accuracy_mean = []
    train_accuracy_minus = []

    test_accuracy_plus = []
    test_accuracy_mean = []
    test_accuracy_minus = []

    for T in params:
        train_mean = np.array(fold_results_train[T]).mean()
        train_var = np.array(fold_results_train[T]).std() / len(fold_results_train[T])

        train_accuracy_plus += [train_mean + train_var]
        train_accuracy_mean += [train_mean]
        train_accuracy_minus += [train_mean - train_var]

    if log_scale:
        params = np.log10(params)

    fig = plt.figure(facecolor = 'w', figsize = (12, 12))
    ax = plt.subplot(111)

    plt.plot(params, train_accuracy_plus, 'r+-')
    plt.plot(params, train_accuracy_mean, 'b', label = "Training Accuracy")
    plt.plot(params, train_accuracy_minus, 'r--')

    if log_scale:
        plt.xlabel('Param, log scale')
    else:
        plt.xlabel('Param')

    plt.ylabel('accuracy')
    plt.title(label +' Cross Validation')
    plt.legend(loc="upper left")
    if save:
        plt.savefig(label.replace(" ", "_") + "_cross_validate.png")
    else:
        plt.show()

def cross_validate_adaboost_T(X_train, y_train, Ts, power=None, C=None, version=0, doPrint=True):
    fold_results_train = {}
    for T in Ts:
        fold_results_train[T] = []

    kf = KFold(n = X_train.shape[0], n_folds=10, shuffle=True)

    for train_index, test_index in kf:
        my_adaboost = adaboost.Adaboost(Ts[-1], power, C, version)
        index_dic = X_train.index.values
        train_index, test_index = index_dic[train_index], index_dic[test_index]
        my_adaboost.fit(X_train.ix[train_index], y_train[train_index])
        for T in Ts:
            fold_results_train[T] += [my_adaboost.score(X_train.ix[test_index], y_train[test_index], T)]

    max_mean_minus_std = 0
    best_T = 0

    for T in Ts:
        np_array = np.array(fold_results_train[T])
        mean_minus_std = np_array.mean() - np_array.std()
        if mean_minus_std > max_mean_minus_std:
            max_mean_minus_std = mean_minus_std
            best_T = T
    if best_T == Ts[-1] or best_T == Ts[0]:
        print " The cross validation is not effective, best T found on border"
    if doPrint:
        plot_cross_validation_curve(fold_results_train, Ts, "Adaboost", False, False)

    return best_T, max_mean_minus_std

def cross_validate_adaboost_T_C(X_train, y_train, Ts, power, Cs, version=0, doPrint=True):
    fold_results_train = {}

    for C in Cs:
        fold_results_train[C] = {}
        for T in Ts:
            fold_results_train[C][T] = []

    kf = KFold(n = X_train.shape[0], n_folds = 10, shuffle = True)

    for C in Cs:

        #print " C : {0}".format(C)

        for train_index, test_index in kf:

            index_dic = X_train.index.values
            train_index, test_index = index_dic[train_index], index_dic[test_index]

            my_adaboost = adaboost.Adaboost(Ts[-1], power, C, version)
            my_adaboost.fit(X_train.ix[train_index], y_train[train_index])

            for T in Ts:
                fold_results_train[C][T] += [my_adaboost.score(X_train.ix[test_index], y_train[test_index], T)]

    max_mean_minus_std = 0
    best_C = 0
    best_T = 0

    for C in Cs:
        for T in Ts:
            np_array = np.array(fold_results_train[C][T])
            mean_minus_std = np_array.mean() - np_array.std()
            if mean_minus_std > max_mean_minus_std:
                max_mean_minus_std = mean_minus_std
                best_C = C
                best_T = T

    if best_C == Cs[-1] or best_C == Cs[0] or best_T == Ts[-1] or best_T == Ts[0]:
        print " The cross validation is not effective, best C or T found on border"

    if doPrint:
        for C in Cs:
            plot_cross_validation_curve(fold_results_train[C], Ts, "Adaboost C : {0}".format(C), False, False)
    return best_C, best_T, max_mean_minus_std

def cross_validate_rf(X_train, y_train, Rounds, Iterations, Weighted):
    fold_results_train = {}

    for Round in Rounds:
        fold_results_train[Round] = {}
        for Iteration in Iterations:
            fold_results_train[Round][Iteration] = []

    kf = KFold(n = X_train.shape[0], n_folds = 3, shuffle = True)

    for Round in Rounds:

        #print " C : {0}".format(C)

        for train_index, test_index in kf:

            index_dic = X_train.index.values
            train_index, test_index = index_dic[train_index], index_dic[test_index]

            rf = random_forest_adaboost.RandomForestAdaboost(Round, Iterations[-1], Weighted)
            rf.fit(X_train.ix[train_index], y_train[train_index])

            for Iteration in Iterations:
                fold_results_train[Round][Iteration] += [rf.score(X_train.ix[test_index], y_train[test_index], Iteration)]

    max_mean_minus_std = 0
    best_Round = 0
    best_Iteration = 0

    for Round in Rounds:
        for Iteration in Iterations:
            np_array = np.array(fold_results_train[Round][Iteration])
            mean_minus_std = np_array.mean() - np_array.std()
            if mean_minus_std > max_mean_minus_std:
                max_mean_minus_std = mean_minus_std
                best_Round = Round
                best_Iteration = Iteration

    if best_Round == Rounds[-1] or best_Round == Rounds[0] or best_Iteration == Iterations[-1] or best_Iteration == Iterations[0]:
        print " The cross validation is not effective, best Round or Iteration found on border"

    return best_Round, best_Iteration, max_mean_minus_std
