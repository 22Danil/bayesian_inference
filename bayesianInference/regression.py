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

    def makeData(size:int = 50000, features:int = 2, informative:int = 2, targets:int = 1):
        """
            Генерує дані для регерсії
            Повертає масив ознак та цільову змінну, повертає нулі при некоректному введенні
        """
        try:
            features, targetVariables = make_regression(n_samples=size, n_features=features, n_informative=informative, n_targets=targets, bias=0.0, random_state=42)
            return features, targetVariables
        except Exception :
            print("Некоректні дані")
            return 0, 0
        
    def distribution(targetVariables):
        """
            Виводить графік оцінки щільності розподілу, з параметрами μ і σ: мю - математичне очікування, сигма - середньоквадратичне відхилення.
        """
        mu = targetVariables.mean()
        print("μ (мю): ", '%.3f'%(mu))
        sigma = targetVariables.std()
        print("σ (сигма): ", '%.3f'%(sigma))
        norm_rv = sts.norm(loc=mu, scale=sigma)

        df = pd.DataFrame(targetVariables, columns=['Емпірична оцінка'])
        ax = df.plot(kind='density')

        x = np.linspace(-400,400,100)
        pdf = norm_rv.pdf(x)
        plt.plot(x, pdf, label='Теоретична оцінка', alpha=0.5)
        plt.legend()
        plt.ylabel('$f(x)$')
        plt.xlabel('$x$')
        plt.show()
        return 1


