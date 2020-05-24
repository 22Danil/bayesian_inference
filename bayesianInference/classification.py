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
import traceback

class Classification:
    """
        Class. Навчання, побудова графіків
    """
    def __init__(self, features, targetVariables):
        """
            Constructor. features - вибірка ознак, targetVariables - цільова змінна, TRAIN_SIZE - розмір вибірки для навчання
        """
        self.features = features
        self.targetVariables = targetVariables
        self.TRAIN_SIZE = 0.7
        self.trainTime = 0

    def training(self):
        """
            Метод навчає модель по даних що знаходятся у атрибутах
            Дані розбиваються на тестові та тренувальні
            Метод повертає: модель, час навчання, точність при кросвалідації, точність на тестовій вибірці
        """
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

        plt.title('Передбачення навченої моделі')
        plt.scatter(feature_test[:, 0], feature_test[:, 1], c=target_pred)
        plt.show()
        
        return Bayes, self.trainTime, score_cross_val, score_train
        
    def makeData(size:int = 50000, features:int = 2, informative:int = 2, redundant:int = 0, classes:int = 4, clusters_per_class:int = 1, sep:int = 2):
        """
            Генерує дані для класифікації
            Повертає масив ознак та цільову змінну, повертає нулі при некоректному введенні
        """
        try:
            features, targetVariables = make_classification(n_samples=size, n_features=features, n_informative=informative, n_redundant = redundant, n_classes=classes, n_clusters_per_class=clusters_per_class, class_sep=sep)
            plt.scatter(features[:, 0], features[:, 1], c=targetVariables)
            plt.title('Згенерована вибірка')
            plt.show()
            return features, targetVariables
        except ValueError as e:
            print("Кількість інформативних, надлишкових і дублюючих  функцій має бути меншою, ніж загальна кількість функцій!")
            return 0, 0


