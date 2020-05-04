import bayesianInference.regression as BI
#import bayesianInference.classification as classif


#print(len(classif.Regression.makeData()))


#print(len(BI.Regression.makeData()))


#a = BI.Regression()

#x, y = a.makeData()BayesRidge, self.trainTime, score_cross_val, score_train


try:
    x, y = BI.Regression.makeData()
    #BI.Regression.distribution(y)
    model = BI.Regression(x, y)
    mod, time_tr, score_cross, score_train = model.training()
    print(mod)
    print(time_tr)
    print(score_cross)
    print(score_train)
except Exception :
    print(Exception)
    #print(y)




#print(a.test())
