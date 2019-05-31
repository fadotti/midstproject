import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn import svm as _svm 
from sklearn.preprocessing import StandardScaler 
import datetime
import random
import time	 

voce = np.genfromtxt('datasets/voce.csv',delimiter=',',skip_header=1,dtype=np.character)
voce = voce[:,2:]
X = voce[:,:-1].astype(float)
y = voce[:,-1]
y[y==b'"male"']=1
y[y==b'"female"']=-1
y = y.astype(int)

p = svm(X,y,X,y)
p.train(reduction=False)
cose = _svm.SVC(kernel='linear')
k = cose.fit(X,y)

y_pred = k.predict(X)
confusion_matrix = np.matrix([[0, 0], [0, 0]])
current_row = 0
for i in range(len(y_pred)):
    confusion_matrix[int((y[i]+1)/2),int((y_pred[i]+1)/2)]=confusion_matrix[int((y[i]+1)/2),int((y_pred[i]+1)/2)]+1
    current_row += 1
correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
confusion_matrix
correct_classification