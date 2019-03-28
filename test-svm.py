#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn.datasets.samples_generator import make_blobs
 
 
###################################################################################
 
#TRAINING SET SIMULATO
#(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=40)#linearmente separabili 100% (cambio std)
(X,y) = make_blobs(n_samples = 10000, n_features = 2, centers = 2, cluster_std = 1, random_state=12)
 
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
(Xt, yt) = make_blobs(n_samples=3000, n_features = 2, centers=2, cluster_std=1, random_state=12)
 
X1t=np.c_[np.ones((Xt.shape[0])),Xt]
plt.scatter(X1t[:, 1],X1t[:, 2], marker='o', c=yt)
plt.axis([minx,maxx,miny,maxy])
plt.show()
 
#trasformazioni necessarie
X = np.asmatrix(X)
y[y==0]=-1
Xt = np.asmatrix(Xt)
yt[yt==0]=-1
 
#percettrone
p = svm(X, y, Xt, yt)
p.train()
print(p.test()[0])
print(p.test()[1])
print(p.weights)
