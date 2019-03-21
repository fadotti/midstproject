from perceptron import perceptron
import pandas as pd

df = pd.read_csv('heart.csv',sep=',',header=0)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
labels_num = [ -1 if x == 0 else 1 for x in y]
test1 = perceptron(training_data=X,training_labels=labels_num,max_rounds=50000,learn_rate=0.05)

test1.train()
print(test1.weights)


#test1.predict(test1.training_data)


