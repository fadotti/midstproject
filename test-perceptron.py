from perceptron import perceptron
import pandas as pd
import numpy as np

X = pd.read_csv('weather_data.csv',sep=',',header=0)
y = pd.read_csv('weather_labels.csv',sep=',',header=0)
sample = np.random.rand(len(y)) < 0.6
X_train = X.iloc[sample,1:]
X_test = X.iloc[~sample,1:]
y_train = y.iloc[sample,1:]
y_test = y.iloc[~sample,1:]
test1 = perceptron(training_data=X_train,training_labels=y_train,test_data=X_test,test_labels=y_test,max_rounds=1000,learn_rate=0.05)

print(test1.train())
print('-----------')
print(test1.test())



#test1.predict(test1.training_data)


