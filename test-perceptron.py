import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.perceptron import perceptron
from sklearn.datasets.samples_generator import make_blobs


###################################################################################

#TRAINING SET SIMULATO
#(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=40)#linearmente separabili 100% (cambio std)
(X,y) = make_blobs(n_samples = 10000, n_features = 2, centers = 2, cluster_std = 3, random_state=93)
minx=float(np.squeeze(np.array(np.amin(X, 0)))[0])
maxx=float(np.squeeze(np.array(np.amax(X, 0)))[0])
miny=float(np.squeeze(np.array(np.amin(X, 0)))[1])
maxy=float(np.squeeze(np.array(np.amax(X, 0)))[1])
plt.scatter(X[:, 0], X[:,1], marker='o', c=y)
plt.axis([minx,maxx,miny,maxy])
plt.show()

#TEST SET SIMULATO
#(X,y) = make_blobs(n_samples=3000, n_features=2, centers=2, cluster_std=1.05, random_state=40)#linearmente separabili 100%
(Xt, yt) = make_blobs(n_samples=1000, n_features = 2, centers=2, cluster_std=1, random_state=93)
plt.scatter(Xt[:, 0],Xt[:, 1], marker='o', c=yt)
plt.axis([minx,maxx,miny,maxy])
plt.show()

#trasformazioni necessarie
X = np.asmatrix(X)
y[y==0]=-1
Xt = np.asmatrix(Xt)
yt[yt==0]=-1

#percettrone
p = perceptron(X, y, Xt, yt)
p.train('seq')
print(p.test()[0])
print(p.test()[1])
w=p.weights

#grafico
print(w)
Xt=np.array(Xt)
xx = np.linspace(-15, 15)
a =  -w[0]/w[1]
yy = a*xx + w[2]/w[1]
plt.scatter(Xt[:, 0],Xt[:,1], marker='o', c= yt)
plt.plot(xx,yy)
plt.show()


###################################################################################
'''
#TRAINING SET REALE
df = pd.read_csv('datasets/heart.csv',sep=',',header=0)
X = df.iloc[:, :-1] #tutte le righe e senza l'ultima colonna
y = df.iloc[:, -1] #tutte le righe solo l'ultima colonna
labels_num = [ -1 if x == 0 else 1 for x in y]
test1 = perceptron(training_data=X, training_labels=labels_num, test_data=X, test_labels=labels_num, max_rounds=1000000, learn_rate = 1)
#il learning rate è superfluo perchè moltiplicare i pesi per una qualsiasi costante semplicementer iscala i pesi senza cambiarne il segno

test1.train("rand")
print(test1.test()[0])
print(test1.test()[1])
'''
###################################################################################

'''
X = pd.read_csv('datasets/weather_data.csv',sep=',',header=0)
y = pd.read_csv('datasets/weather_labels.csv',sep=',',header=0)

sample = np.random.rand(len(y)) < 0.75
X_train = X.iloc[sample, 1:]
X_test = X.iloc[~sample, 1:]
y_train = y.iloc[sample, 1:]
y_test = y.iloc[~sample, 1:]
'''