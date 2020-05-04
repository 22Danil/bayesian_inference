from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, linear_model

import matplotlib.pyplot as plt
import scipy.stats as sts
import pandas as pd
import numpy as np
import time


class Regression:
    def makeData(size:int = 500):
        """Класс Круг.Конструктор принимает радиус."""
        X, Y = make_regression(n_samples=size, n_features=2, n_informative=2, n_targets=1, bias=0.0, random_state=42)
        return X


