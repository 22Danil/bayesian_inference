from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import matplotlib.pyplot as plt
import scipy.stats as sts
import pandas as pd
import numpy as np
import time


class Classification:
    """Class. Обучение, построение графиков"""
    def __init__(self, features, targetVariables):
        """Constructor. features - выборка признаков, targetVariables - целевая переменаня"""
        self.features = features
        self.targetVariables = targetVariables
        self.TRAIN_SIZE = 0.7
        self.trainTime = 0

    def training(self):
        feature_train, feature_test, target_train, target_test = train_test_split(self.features, self.targetVariables, train_size=self.TRAIN_SIZE, random_state=3)
        
        Bayes = GaussianNB()
        startTime = time.time()
        Bayes.fit(feature_train, target_train)
        stopTime = time.time()
        self.trainTime = stopTime - startTime
        
        score = cross_val_score(Bayes, feature_train, target_train, cv = 10, scoring="accuracy")
        score_cross_val = score.mean()
        target_pred = Bayes.predict(feature_test)
        score_train = metrics.accuracy_score(target_test, target_pred)

        plt.title('Сгенерированная выборка')
        plt.scatter(feature_test[:, 0], feature_test[:, 1], c=target_pred)
        plt.show()
        
        return Bayes, self.trainTime, score_cross_val, score_train
        
    def makeData(size:int = 50000):
        """Генерация выборки для классификации."""
        try:
            features, targetVariables = make_classification(n_samples=size, n_features=2, n_informative=2, n_redundant = 0, n_classes=4, n_clusters_per_class=1, class_sep=2)
            plt.scatter(features[:, 0], features[:, 1], c=targetVariables)
            plt.title('Сгенерированная выборка')
            plt.show()
            return features, targetVariables
        except Exception :
            #print("You cannot divide by zero!")
            return 0, 0


