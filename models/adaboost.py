import numpy as np
from sklearn.tree import DecisionTreeClassifier
import copy

class Adaboost:
    """
    D : Dictionnary of weights D[i] = list of m weights
    """
    D = np.empty(0)

    """
    alpha_list : List of alphas of the Adaboost best h
    """
    alpha_list = np.empty(0)

    """
    h_list : List of h_t best classifiers selected
    """
    h_list = np.empty(0)

    """
    T : number of iterations
    """
    T = 0

    """
    decisionTreeClassifier : Decision tree classifier
    """
    decisionTreeClassifier = 0

    """
    regularization_power :  None if no regularization else a positive integer.
    -> Regularize with a (Sum(alpha_t D_t(i)) power regularization)
    """
    regularization_power = None

    """
    regularization_C :  None if no regularization else a positive integer.
    -> Regularize with a (Sum(alpha_t D_t(i)) power regularization)
    """
    regularization_C = 1

    """
    Version : We try multiple regularizations
    - version 0 : base regularization : Weight update -> C * (Psi_t-1(i) - Psi_t(i))
    - version 1 : Updated : Weight update -> - C * Psi_t(i)
    """
    version = 0

    """
    Random : Randomize weights at the beginning
    """
    random = False

    def __init__(self, T, regularization_power=None, regularization_C=None, version=0, random=False):
        self.version = version
        self.regularization_power = regularization_power
        self.regularization_C = regularization_C
        self.D = np.empty(0)
        self.alphas = np.empty(0)
        self.h_t = np.empty(0)
        self.T = T
        self.random = random
        self.decisionTreeClassifier = DecisionTreeClassifier(criterion="entropy", max_depth=1, presort=True)

    def fit(self, trainFeatures, trainLabels):
        error_rate = []
        feat = []

        """
        fit Adaboost to the training set, with T iterations.
        """

        m, n = trainFeatures.shape

        self.D = np.ndarray(shape=(self.T + 1, m))

        if self.random:
            self.D.fill( 1.0 / m)
            self.D[0] = np.random.rand(m)
            self.D[0] /= self.D[0].sum()
        else:
            self.D.fill(1.0 / m)

        psi_t = np.ndarray(shape=(self.T, m))

        for t in range(self.T):
            h_t = self.decisionTreeClassifier.fit(trainFeatures, trainLabels, self.D[t])

            h_t_predict_all = h_t.predict(trainFeatures)
            epsilon_t =  np.where(h_t_predict_all != trainLabels, 1.0, 0) * self.D[t]
            epsilon_t = epsilon_t.sum()

            alpha_t = np.log((1 - epsilon_t) / epsilon_t) / 2

            if epsilon_t == 0:
                alpha_t = 1

            self.alpha_list = np.append(self.alpha_list, alpha_t)

            regularization_vector = np.zeros(m)

            if self.regularization_power != None:
                regularization_vector -=  np.power(np.transpose(self.D[:t+1]).dot(self.alpha_list[:t+1] / np.linalg.norm(self.alpha_list[:t+1])), self.regularization_power)
                psi_t[t] = - regularization_vector
                if t > 0 and self.version == 0:
                    regularization_vector += np.power(np.transpose(self.D[:t]).dot(self.alpha_list[:t]/ np.linalg.norm(self.alpha_list[:t])), self.regularization_power)
                regularization_vector *= self.regularization_C

            self.D[t + 1] =  self.D[t] * np.exp(- alpha_t * trainLabels * h_t_predict_all + regularization_vector)

            if self.regularization_power != None:
                self.D[t + 1] /= np.linalg.norm(self.D[t + 1], ord=1)
            else:
                Z_t = 2 * np.sqrt(epsilon_t * ( 1 - epsilon_t))
                self.D[t + 1] /= Z_t

            if len(self.alpha_list) > 1 and self.alpha_list[-2] == alpha_t:
                print " Stoping the training at t = {0}, convergence.".format(t)
                break

            error_rate += [epsilon_t]
            feat += [np.argmax(h_t.feature_importances_)]

            self.h_list = np.append(self.h_list, copy.copy(h_t))
            if epsilon_t == 0:
                break

        return self.D

    def predict_proba(self, x, T=0):
        """
        Return the g(x), not sgn(g(x))
        """
        if T == 0 or T > len(self.alpha_list):
            T = len(self.alpha_list)

        scores = np.zeros(x.shape[0])
        for i in range(T):
            scores = scores + self.alpha_list[i] * self.h_list[i].predict(x)
        return scores

    def predict(self, x, T=0):
        """
        Once trained, predict a label given x
        T : if we compute adaboost(1000) and we want adaboost(10), it's the 10 first terms of adaboost(1000). Let's use it.
        """
        if T == 0 or T > len(self.h_list):
            T = len(self.h_list)

        scores = np.zeros(x.shape[0])

        for i in range(T):
            scores = scores + self.alpha_list[i] * self.h_list[i].predict(x)
        return np.where(scores > 0, 1, -1)

    def error(self, x, y, T):
        """
        Return the percentage of errors in the prediction.
        """
        return np.where(self.predict(x, T) * y == 1, 0, 1).sum() * 1.0 / x.shape[0]

    def score(self, x, y, T = None):
        if T == None or T > self.T:
            T = self.T
        return np.where(self.predict(x, T) * y == 1, 1, 0).sum() * 1.0 / x.shape[0]

    def toString(self):
        print "T : {0}, Alphas : {0}".format(self.T, len(self.alpha_list))
