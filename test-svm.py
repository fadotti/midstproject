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
(X,y) = make_blobs(n_samples = 1000000, n_features = 2, centers = 2, cluster_std =1, random_state=99)



""" X1=np.c_[np.ones((X.shape[0])),X]
minx=float(np.squeeze(np.array(np.amin(X, 0)))[0])
maxx=float(np.squeeze(np.array(np.amax(X, 0)))[0])
miny=float(np.squeeze(np.array(np.amin(X, 0)))[1])
maxy=float(np.squeeze(np.array(np.amax(X, 0)))[1])
plt.scatter(X[:, 0], X[:,1], marker='o', c=y)
plt.axis([minx,maxx,miny,maxy])
plt.show() """
 
#TEST SET SIMULATO
#(X,y) = make_blobs(n_samples=3000, n_features=2, centers=2, cluster_std=1.05, random_state=40)#linearmente separabili 100%
(Xt, yt) = make_blobs(n_samples=3000, n_features = 2, centers=2, cluster_std=1, random_state=99)
 
X1t=np.c_[np.ones((Xt.shape[0])),Xt]
""" plt.scatter(X1t[:, 1],X1t[:, 2], marker='o', c=yt)
plt.axis([minx,maxx,miny,maxy])
plt.show()
 """ 
#trasformazioni necessarie
X = np.asmatrix(X)
y[y==0]=-1
Xt = np.asmatrix(Xt)
yt[yt==0]=-1
#standardizzo i dati sia di training che di set
X = scale.fit_transform(X)
Xt = scale.fit_transform(Xt)
p = svm(X, y, Xt, yt,C=1,max_rounds=100)
time.sleep(1)

# segna il momento d'inizio
starttime = datetime.datetime.now()
p.train(0.1)
endtime = datetime.datetime.now()
# calcola il tempo trascorso
deltaT = endtime - starttime
# calcola il tempo medio
accessTime = deltaT.total_seconds() * 1000
print(accessTime)
w=p.weights

#p2.train()
#w2 = p2.weights
cose = _svm.SVC(kernel='linear')
k = cose.fit(X,y)
xx = np.linspace(-2.5, 2.5)
X1_std= scale.fit_transform(X1)
""" a =  -w[0]/w[1]
yy = a*xx
#per scikitsvm
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx

#plot delle due rette
plt.plot(xx,yy)
plt.plot(xx,yy1)

#support vectors
plt.scatter(X[sv,0],X[sv,1], c= 'b') #support vector mysvm
plt.scatter(k.support_vectors_[:,0],k.support_vectors_[:,1], c= 'r') #support vector scikitsvm
plt.show()





########################################################


print('--------------------')
print(k.support_)
print('--------------------')
print(sv)


#print(sv)
#mysvm1 = svm(X[sv], y[sv], Xt, yt,C=1,max_rounds=100)
#mysvm1.train()

#se uso i vettori di supporto di scikit la retta è comunque simile alla originale quindi boh
mysvm2 = svm(X[k.support_],y[k.support_], Xt, yt,C=1,max_rounds=100)
mysvm2.train()
w2 = mysvm2.weights
print(w2) #pesi

xx = np.linspace(-2.5, 2.5)
plt.scatter(X[:, 0], X[:,1], marker='o', c=y) """

#coefficienti delle rette
#per mysvm
a =  -w[0]/w[1]
yy = a*xx
#per scikitsvm
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx
a =  -w[0]/w[1]
#a2 =  -w2[0]/w2[1]
yy = a*xx
#yy2 = a2*xx
plt.scatter(Xt[:, 0], Xt[:,1], marker='o', c=yt)
plt.plot(xx,yy,color='blue')
plt.plot(xx,yy1,color='orange')
#plt.plot(xx,yy2,color='purple')
plt.plot(xx, yy1+(1-k.intercept_[0]/np.sqrt(np.sum(k.coef_[0]**2))),linestyle='dashed',color='orange')
plt.plot(xx, yy1-(1-k.intercept_[0]/np.sqrt(np.sum(k.coef_[0]**2))),linestyle='dashed',color='orange')
plt.plot(xx, yy+(1-w[2]/np.sqrt((w[0]+w[1])**2)),linestyle='dashed',color='blue')
plt.plot(xx, yy-(1-w[2]/np.sqrt((w[0]+w[1])**2)),linestyle='dashed',color='blue')
#plt.plot(xx, yy2+(1-w2[2]),linestyle='dashed',color='purple')
#plt.plot(xx, yy2-(1-w2[2]),linestyle='dashed',color='purple')
# plt.scatter(k.support_vectors_[:,0],k.support_vectors_[:,1])
#plt.scatter(np.squeeze(np.array(p.training_data[list(p.support_vectors),0])),np.squeeze(np.array(p.training_data[list(p.support_vectors),1])))
plt.show()

#########################################################################

