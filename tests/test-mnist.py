import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn import svm as _svm  


train = np.genfromtxt('./datasets/mnist/mnist_train.csv',delimiter=',',skip_header=1)
print({i:len(train[train[:,0]==i,:]) for i in set(train[:,0])})

test = np.genfromtxt('./datasets/mnist/mnist_test.csv',delimiter=',',skip_header=1)

print({i:len(test[test[:,0]==i,:]) for i in set(test[:,0])})

# scegliamo 3 e 9 (possono essere due cifre qualsiasi)
 
binary_train = np.array([el for el in train if el[0]==1 or el[0]==7])
print({i:len(binary_train[binary_train[:,0]==i,:]) for i in set(binary_train[:,0])})
y = binary_train[:,0]
X = binary_train[:,1:]

binary_test = np.array([el for el in test if el[0]==1 or el[0]==7])
y_test = binary_test[:,0]
X_test = binary_test[:,1:]
print({i:len(binary_test[binary_test[:,0]==i,:]) for i in set(binary_test[:,0])})
"""
y=train[:,0]
X=train[:,:-1]
y[y==0]=1
y[y==1]=1
y[y==2]=1
y[y==3]=1
y[y==4]=1
y[y!=1]=-1
y_test=test[:,0]
X_test=test[:,:-1]
y_test[y_test==0]=1
y_test[y_test==1]=1
y_test[y_test==2]=1
y_test[y_test==3]=1
y_test[y_test==4]=1
y_test[y_test!=1]=-1
""" 
y_test[y_test==1]=1
y_test[y_test==7]=-1

mysvm = svm(X,y,X_test,y_test)
coeff = mysvm.train()
mysvm.test()

cose = _svm.SVC(kernel='linear')
k = cose.fit(X,y)
np.sum((mysvm.weights[:-1]-k.coef_)**2)

y_pred = k.predict(X_test)
confusion_matrix = np.matrix([[0, 0], [0, 0]])
current_row = 0
for i in range(len(y_pred)):
    confusion_matrix[int((y_test[i]+1)/2),int((y_pred[i]+1)/2)]=confusion_matrix[int((y_test[i]+1)/2),int((y_pred[i]+1)/2)]+1
    current_row += 1
correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
confusion_matrix
correct_classification