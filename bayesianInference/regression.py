from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics, linear_model

import matplotlib.pyplot as plt
import scipy.stats as sts
import pandas as pd
import numpy as np
import time


class Regression:
    """Class. Обучение, построение графиков"""
    def __init__(self, features, targetVariables):
        """Constructor. features - выборка признаков, targetVariables - целевая переменаня"""
        self.features = features
        self.targetVariables = targetVariables
        self.TRAIN_SIZE = 0.7
        self.trainTime = 0
        
    def training(self):
        feature_train, feature_test, target_train, target_test = train_test_split(self.features, self.targetVariables, train_size=self.TRAIN_SIZE, random_state=3)
        
        BayesRidge = linear_model.BayesianRidge()
        startTime = time.time()
        BayesRidge.fit(feature_train, target_train)
        stopTime = time.time()
        self.trainTime = stopTime - startTime
        
        score = cross_val_score(BayesRidge, feature_train, target_train, cv = 10, scoring="r2")
        score_cross_val = score.mean()
        target_pred = BayesRidge.predict(feature_test)
        score_train = metrics.r2_score(target_test, target_pred)
        return BayesRidge, self.trainTime, score_cross_val, score_train

    def makeData(size:int = 50000, features:int = 2, informative:int = 2):
        """Генерация выборки для регрессии."""
        try:
            features, targetVariables = make_regression(n_samples=size, n_features=features, n_informative=informative, n_targets=1, bias=0.0, random_state=42)
            return features, targetVariables
        except Exception :
            #print("You cannot divide by zero!")
            return 0, 0
        
    def distribution(targetVariables):
        """Возвращает плотность вероятности целевой переменной"""
        mu = targetVariables.mean()
        print("МЮ - ", mu)
        sigma = targetVariables.std()
        print("Sigma - ", sigma)
        norm_rv = sts.norm(loc=mu, scale=sigma)

        df = pd.DataFrame(targetVariables, columns=['KDE'])
        ax = df.plot(kind='density')

        x = np.linspace(-400,400,100)
        pdf = norm_rv.pdf(x)
        plt.plot(x, pdf, label='theoretical pdf', alpha=0.5)
        plt.legend()
        plt.ylabel('$f(x)$')
        plt.xlabel('$x$')
        plt.show()
        return 1


