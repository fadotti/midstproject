import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm as _svm 
from sklearn.preprocessing import StandardScaler 

###################################################################################

scale = StandardScaler()

#TRAINING SET SIMULATO
(X,y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1, random_state=18)
#assi del grafico + grafico training set
minx=float(np.squeeze(np.array(np.amin(X, 0)))[0])
maxx=float(np.squeeze(np.array(np.amax(X, 0)))[0])
miny=float(np.squeeze(np.array(np.amin(X, 0)))[1])
maxy=float(np.squeeze(np.array(np.amax(X, 0)))[1])
plt.scatter(X[:, 0], X[:,1], marker='o', c=y)
plt.axis([minx,maxx,miny,maxy])
plt.show()
 
#TEST SET SIMULATO
(Xt, yt) = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1, random_state=18)
#grafico test set
plt.scatter(Xt[:, 0],Xt[:, 1], marker='o', c=yt)
plt.axis([minx,maxx,miny,maxy])
plt.show()
 
#trasformazioni necessarie per passare i due set alle funzioni di svm
X = np.asmatrix(X)
y[y==0]=-1
Xt = np.asmatrix(Xt)
yt[yt==0]=-1
#standardizzo i dati sia di training che di set
X = scale.fit_transform(X)
Xt = scale.fit_transform(Xt)


################### MYSVM ##############################
mysvm = svm(X, y, Xt, yt,C=1,max_rounds=100)
mysvm.train()
w = mysvm.weights
sv = list(mysvm.support_vectors) #vettori di supporto trovati dalla nostra svm
#print(sv)
print(w) #pesi

################### SCIKITSVM ##########################
scikitsvm = _svm.SVC(kernel='linear')
k = scikitsvm.fit(X,y)
sv_indexes = k.support_ #indici dei vettori di supporto trovati da scikit svm
#print(sv_indexes)
print(k.coef_,k.intercept_)#pesi


'''
#vedo quali stanno effettivamente sui piani pari a 1 e -1 alla fine dello sgd
for sv in k.support_vectors_:
    x = np.array(sv)
    w = np.array(k.coef_)
    b = k.intercept_
    print(sv, np.dot(w,x) + b) #dovrebbe essere pari a 1
#solo due punti stano effettivamente sul piano 

#rifaccio svm con solo i support vectors e ottengo gli stessi pesi e bias
scikitsvm1 = _svm.SVC(kernel='linear')
kk = scikitsvm1.fit(X[k.support_],y[k.support_])
print(kk.coef_,k.intercept_)
'''


################### GRAFICI #############################

xx = np.linspace(-2.5, 2.5)
plt.scatter(X[:, 0], X[:,1], marker='o', c=y)

#coefficienti delle rette
#per mysvm
a =  -w[0]/w[1]
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
plt.scatter(X[:, 0], X[:,1], marker='o', c=y)

#coefficienti delle rette
#per mysvm
a =  -w[0]/w[1]
yy = a*xx
#per scikitsvm
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx

a2 =  -w2[0]/w2[1]
yy2 = a2*xx

#plot delle due rette
plt.plot(xx,yy)
plt.plot(xx,yy1)
plt.plot(xx,yy2)
plt.show()