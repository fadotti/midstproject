#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm as _svm 
from sklearn.preprocessing import StandardScaler 
###################################################################################
scale = StandardScaler()
#TRAINING SET SIMULATO
#(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=62)#linearmente separabili 100% (cambio std)
(X,y) = make_blobs(n_samples = 100000, n_features = 2, centers = 2, cluster_std = 1, random_state=99)



X1=np.c_[np.ones((X.shape[0])),X]
minx=float(np.squeeze(np.array(np.amin(X, 0)))[0])
maxx=float(np.squeeze(np.array(np.amax(X, 0)))[0])
miny=float(np.squeeze(np.array(np.amin(X, 0)))[1])
maxy=float(np.squeeze(np.array(np.amax(X, 0)))[1])
plt.scatter(X1[:, 1], X1[:,2], marker='o', c=y)
plt.axis([minx,maxx,miny,maxy])
plt.show()
 
#TEST SET SIMULATO
#(X,y) = make_blobs(n_samples=3000, n_features=2, centers=2, cluster_std=1.05, random_state=40)#linearmente separabili 100%
(Xt, yt) = make_blobs(n_samples=3000, n_features = 2, centers=2, cluster_std=1, random_state=99)
 
X1t=np.c_[np.ones((Xt.shape[0])),Xt]
plt.scatter(X1t[:, 1],X1t[:, 2], marker='o', c=yt)
plt.axis([minx,maxx,miny,maxy])
plt.show()
 
#trasformazioni necessarie
X = np.asmatrix(X)
y[y==0]=-1
Xt = np.asmatrix(Xt)
yt[yt==0]=-1
X = scale.fit_transform(X)
Xt = scale.fit_transform(Xt)
p = svm(X, y, Xt, yt,C=1,max_rounds=100)

p.train()
w=p.weights
cose = _svm.SVC(kernel='linear')
k = cose.fit(X,y)
""" xx = np.linspace(-2.5, 2.5)
X1_std= scale.fit_transform(X1)
a =  -w[0]/w[1]
yy = a*xx
plt.scatter(X1_std[:, 1], X1_std[:,2], marker='o', c=y)
plt.plot(xx,yy)
plt.plot(xx,yy+(1-w[2]),linestyle='dashed',color='red')
plt.plot(xx,yy-(1-w[2]),linestyle='dashed',color='red')
plt.show() """




Xt_std= scale.fit_transform(X1t)
xx = np.linspace(-2.5, 2.5)
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx
w = w*np.sqrt(np.sum(w**2))
a =  -w[0]/w[1]
yy = a*xx
plt.scatter(Xt_std[:, 1], Xt_std[:,2], marker='o', c=yt)
plt.plot(xx,yy)
plt.plot(xx,yy1)
plt.plot(xx, yy1+(1-k.intercept_[0]/np.sqrt(np.sum(k.coef_[0]**2))),linestyle='dashed',color='orange')
plt.plot(xx, yy1-(1-k.intercept_[0]/np.sqrt(np.sum(k.coef_[0]**2))),linestyle='dashed',color='orange')
plt.plot(xx, yy+(1-w[2]),linestyle='dashed',color='red')
plt.plot(xx, yy-(1-w[2]),linestyle='dashed',color='red')
plt.scatter(k.support_vectors_[:,0],k.support_vectors_[:,1])
plt.show()