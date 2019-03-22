from perceptron import perceptron
import pandas as pd

X = pd.read_csv('weather_data.csv',sep=',',header=0)
y = pd.read_csv('weather_labels.csv',sep=',',header=0)
test1 = perceptron(training_data=X,training_labels=y,max_rounds=1000,learn_rate=0.05)

test1.train()
print(test1.weights)


#test1.predict(test1.training_data)


