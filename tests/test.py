import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm as _svm 
from sklearn.preprocessing import StandardScaler 
import datetime
import random
import time	 
###################################################################################

scale = StandardScaler()

#TRAINING SET SIMULATO
#(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=62)#linearmente separabili 100% (cambio std)
(X,y) = make_blobs(n_samples = 5000, n_features = 2, centers = 2, cluster_std =1, random_state=0)
(X,y) = make_blobs(n_samples = 5000, n_features = 2, centers = 2, cluster_std =1, random_state=99)
(Xt,yt) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std =1, random_state=99)

#trasformazioni necessarie
X = np.asmatrix(X)
y[y==0]=-1
X = scale.fit_transform(X)
Xt = np.asmatrix(Xt)
yt[yt==0]=-1
Xt = scale.fit_transform(Xt)


plt.scatter(X[:, 0], X[:,1], marker='o', c=y)
plt.show()


#standardizzo i dati sia di training che di set
p = svm(X,y,Xt,yt,C=1,max_rounds=100)
p2 = svm(X,y,Xt,yt,C=1,max_rounds=100)
p3 = svm(X,y,Xt,yt,C=1,max_rounds=100)
p.train(0.1)
p2.train(0.1,reduction=False)
p3.train(reduction=False)
cose = _svm.SVC(kernel='linear',max_iter=100)
k = cose.fit(X,y)


y_pred = k.predict(Xt)
confusion_matrix = np.matrix([[0, 0], [0, 0]])
current_row = 0
for i in range(len(y_pred)):
    confusion_matrix[int((yt[i]+1)/2),int((y_pred[i]+1)/2)]=confusion_matrix[int((yt[i]+1)/2),int((y_pred[i]+1)/2)]+1
    current_row += 1
correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
p.test()
p2.test()
p3.test()
confusion_matrix
correct_classification


y_prediction = np.array([p.predict(x) for x in p.test_data])


xx = np.linspace(-2.5, 2.5)
#coefficienti delle rette
#per mysvm
a =  -p.weights[0]/p.weights[1]
yy = a*xx
#per scikitsvm
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx
a2 =  -p2.weights[0]/p2.weights[1]
yy2 = a2*xx
a3 = -p3.weights[0]/p3.weights[1]
yy3 = a3*xx

plt.scatter(Xt[:, 0], Xt[:,1], marker='o', c=yt)

plt.scatter(Xt[:, 0], Xt[:,1], marker='+', c=y_prediction*10)
plt.plot(xx,yy,color='blue')
plt.plot(xx,yy1,color='orange')
plt.plot(xx,yy2,color='green')
plt.plot(xx,yy3,color='red')
plt.show()



