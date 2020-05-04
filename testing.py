

'''
import bayesianInference.regression as BI

x, y = BI.Regression.makeData()
BI.Regression.distribution(y)

model = BI.Regression(x, y)
mod, time_tr, score_cross, score_train = model.training()
print("Время обучени - ", '%.5f'%(time_tr))
print("Точность при кроссвалидации - ", '%.4f'%(score_cross))
print("Точность на тестовой выборке - ", '%.4f'%(score_train))
'''



'''
import bayesianInference.classification as NB
x, y = NB.Classification.makeData()
model = NB.Classification(x, y)
mod, time_tr, score_cross, score_train = model.training()
print("Время обучени - ", '%.5f'%(time_tr))
print("Точность при кроссвалидации - ", score_cross)
print("Точность на тестовой выборке - ", score_train)
'''

