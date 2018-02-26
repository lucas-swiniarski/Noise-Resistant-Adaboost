# Change Matplotlib inline, show vs Savefig
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import adaboost

# utils
import utils

# Allow matplotlib to plot on HPC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# For printing on stdout on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

print_results = True
k_fold_number = 10
base_num_iter = 200

print(" Read data ... ")

data = pd.read_csv("data/waveform.data.clean", sep="\t")
X_train, y_train, X_test, y_test = utils.split_rebalance_training_set(data, True, ["Y_value", "potential_money_made", "LOAN_ID"], "Y")

"""
Cross Validating the number of iter before Convergence ...
"""
"""
print(" Cross validating SVM Num Iters...")

NumIters = [100 * i for i in range(1, 21)]
fold_results_train = {}

for NumIter in NumIters:
    print "Num Iter : {0}".format(NumIter)
    svm = SGDClassifier(loss = "hinge", n_iter = NumIter, shuffle = True)
    scores = cross_val_score(svm, X_train, y_train, cv= k_fold_number, n_jobs = -1, scoring="roc_auc")
    fold_results_train[NumIter] = scores

utils.plot_cross_validation_curve(fold_results_train, NumIters, "SVM Num Iters", False, True)
"""

"""
Cross Validating Logistic Regression :
- Alpha
- Number of Iterations
- Penalty term
- learning_rate
"""

print(" Cross validating SVM ...")

Cs = [10 ** i for i in range(-10, 5)]
fold_results_train = {}

for c in Cs:
    print "C : {0}".format(c)
    svm = LinearSVC(C=c, dual=False)
    scores = cross_val_score(svm, X_train, y_train, cv= k_fold_number, n_jobs = -1, scoring="roc_auc")
    fold_results_train[c] = scores

utils.plot_cross_validation_curve(fold_results_train, Cs, "SVM c CV", True, True)
