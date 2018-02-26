import pandas as pd
import numpy as np

"""
This file add noise to a data-set
"""

def addNormalNoise(column, avg, std):
    return column + np.random.normal(avg, std, column.shape[0])

def addUniformNoise(column, low, high, list_features=None):
    return column + np.random.uniform(avg, std, column.shape[0])

def switchLabels(target, pPositive, pNegative):
    """
    Switch labels of positive label ( + 1 ) with probability pPositive,
    and negative labels ( -1 ) with probability pNegative
    """

    for i in target.index.values:
        rand = np.random.rand()
        if (target.ix[i] == 1 and rand < pPositive) or (target.ix[i] == -1 and rand < pNegative):
            target.ix[i] = - target.ix[i]
    return target
