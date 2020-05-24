'''
#Приклад з завантаженням даних

import bayesianInference.regression as BI
import matplotlib.pyplot as plt
import scipy.stats as sts
import pandas as pd
import numpy as np

print("sdsdds")

data=pd.read_csv('winequality-red.csv')

data_split = np.array(data)
data_Y = data_split[:, -1]
data_X = data_split[:, :-1]
model = BI.Regression(data_X, data_Y)
mod, time_tr, score_cross, score_train = model.training()
print("Час навчання: ", '%.5f'%(time_tr))
print("Точність при кроссвалідаціі: ", '%.4f'%(score_cross))
print("Точність на тестовій вибірці: ", '%.4f'%(score_train))
'''

'''
#Регресія

import bayesianInference.regression as BI
x, y = BI.Regression.makeData()
BI.Regression.distribution(y)
model = BI.Regression(x, y)
mod, time_tr, score_cross, score_train = model.training()
print("Час навчання: ", '%.5f'%(time_tr))
print("Точність при кроссвалідаціі: ", '%.4f'%(score_cross))
print("Точність на тестовій вибірці: ", '%.4f'%(score_train))
'''

'''
#Класифікація

import bayesianInference.classification as NB
x, y = NB.Classification.makeData()
model = NB.Classification(x, y)
mod, time_tr, score_cross, score_train = model.training()
print("Час навчання: ", '%.5f'%(time_tr))
print("Точність при кроссвалідаціі: ", '%.4f'%(score_cross))
print("Точність на тестовій вибірці: ", '%.4f'%(score_train))
'''


'''
import bayesianInference.classification as NB
import bayesianInference.regression as BI
print("1 - згенерувати для регресії + розподіл, 2 - згенерувати для класифікації, 3 - навчання (регресія), 4 - навчання(класифікація), 5 - вихід")
logoff = False
x = 0
y = 0
while(not logoff):
    print("Введіть команду:")
    a = input()
    if a == "1":
        x, y = BI.Regression.makeData()
        BI.Regression.distribution(y)
    elif a == "2":
        x, y = NB.Classification.makeData()
    elif a == "3":
        model = BI.Regression(x, y)
        mod, time_tr, score_cross, score_train = model.training()
        print("Час навчання: ", '%.5f'%(time_tr))
        print("Точність при кроссвалідаціі: ", '%.4f'%(score_cross))
        print("Точність на тестовій вибірці: ", '%.4f'%(score_train))
    elif a == "4":
        model = NB.Classification(x, y)
        mod, time_tr, score_cross, score_train = model.training()
        print("Час навчання: ", '%.5f'%(time_tr))
        print("Точність при кроссвалідаціі: ", '%.4f'%(score_cross))
        print("Точність на тестовій вибірці: ", '%.4f'%(score_train))
    elif a == "5":
        logoff = True
'''
