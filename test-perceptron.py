from perceptron import perceptron
import pandas as pd

df = pd.read_csv('iris.dat',sep=',',header=None)
X = df.iloc[:,0:4]
y = df.iloc[:,-1]

labels_num = [ -1 if x == 'Iris-setosa' else 1 for x in y]
test1 = perceptron(training_data=X,training_labels=labels_num)

test1.train()
print test1.weights


#test1.predict(test1.training_data)


