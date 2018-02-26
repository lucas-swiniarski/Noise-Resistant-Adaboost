import numpy as np
import adaboost
import pandas as pd

class RandomForestAdaboost:
    bootstrap_percentage = 2.0/3
    number_feature = 0
    rounds = 0
    list_adaboost_classifier = np.empty(0)
    list_subfeatures = np.empty(0)
    iteration = 0
    weighted = False
    weights = np.empty(0)
    random = False
    trainError = np.empty(0)


    def __init__(self, rounds, iteration, weighted=False, random=False):
        self.rounds = rounds
        self.iteration = iteration
        self.weighted = weighted
        self.random = random
        self.trainError = np.empty(rounds)

    def fit(self, trainFeatures, trainLabels):
        self.number_feature = int(np.sqrt(trainFeatures.shape[1]))
        self.list_subfeatures = np.ndarray(shape=(self.rounds, self.number_feature ))

        for i in range(self.rounds):
            features_selected = np.random.choice(trainFeatures.columns.values, self.number_feature, replace=False)
            bootstraped_index = np.random.choice(trainFeatures.index.values, self.bootstrap_percentage * trainFeatures.shape[0])
            self.list_subfeatures[i] = features_selected
            my_adaboost = adaboost.Adaboost(self.iteration, random=self.random)
            my_adaboost.fit(trainFeatures.ix[bootstraped_index][features_selected], trainLabels.ix[bootstraped_index])
            self.trainError[i] = my_adaboost.error(, y, T)
            self.list_adaboost_classifier = np.append(self.list_adaboost_classifier, [my_adaboost])
            if self.weighted:
                self.weights = np.append(self.weights, [my_adaboost.score(trainFeatures.ix[np.delete(trainFeatures.index.values, bootstraped_index)][features_selected], trainLabels.ix[np.delete(trainFeatures.index.values, bootstraped_index)])])
        return self.weights, self.trainError

    def score(self, testFeatures, testLabels, Iteration=None):
        if Iteration == None:
            Iteration = self.iteration
        prediction = np.zeros(testLabels.shape[0])
        for i in range(self.list_adaboost_classifier.shape[0]):
            if self.weighted:
                prediction += self.weights[i] * self.list_adaboost_classifier[i].predict(testFeatures[self.list_subfeatures[i]], Iteration)
            else:
                prediction += self.list_adaboost_classifier[i].predict(testFeatures[self.list_subfeatures[i]], Iteration)
        return np.where(prediction * testLabels > 0, 1,  0).sum() * 1.0 / testLabels.shape[0]
