import numpy as np
import random as rn
#import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.utils import shuffle

def minimize_stochastic(target_f, gradient_f, X, y, starting_w, starting_l_rate = .01):# se cambio lr la stima cambia abbastanza

    w = starting_w
    min_w, min_f = None, float('inf') #w corrispondente al valore minimo della funzione di costo
    l_rate = starting_l_rate
    no_improv = 0 #conta quante iterazioni senza miglioramenti ci sono state
    x = list(range(X.shape[0]))

    while no_improv < 100:

        #convergenza
        f_value = target_f(X, y, w) #valore della funzione con il dataset e i valori correnti di w
        #print(f_value)
        if f_value < min_f: #se il valore della funzione e minore del valore minimo attuale
            min_w, min_f = w, f_value #aggiorno i pesi e il valore minimo
            no_improv = 0 #reimposto il contatore a 0
        else: #altrimenti
            no_improv += 1 #aggiungo 1 al contatore
            #print(no_improv)
            l_rate *= 0.9 #diminuisco il learning rate

        #aggiornamento dei pesi
        #shuffle del dataset
        #X_1, y_1 = shuffle(X, y) #X e y originali non cambiano
        #data = zip(X_1, y_1)
	
	rn.shuffle(x)

        for i in x:
            #print(row[0])
            gradient_i = gradient_f(X[i], y[i], w)
            #print(gradient_i)
            w = w - l_rate * gradient_i  # aggiorno i pesi

    return min_w, min_f









def cost_function(X, y, w, C = 1): #questa e calcolata su tutte le osservazioni del dataset
    return (1/2) * np.sum(w**2) + C * np.sum(map(lambda xi,yi: max(0, 1 - yi * np.dot(w, xi)),X, y))

def gradient_of_cost(x, yi, w, C = 1): #x e il vettore di esplicative di una singola osservazione
    # x e w sono vettori quindi la funzione ritorna un vettore px1
    return w if (yi * np.dot(w, x)) >= 1 else w - C * (yi * x)






nf = 3 #numero di esplicative
(X,y) = make_blobs(n_samples = 10000, n_features = nf, centers = 2, cluster_std = 1, random_state=40)
X1=np.c_[np.ones((X.shape[0])),X]
#minx=float(np.squeeze(np.array(np.amin(X, 0)))[0])
#maxx=float(np.squeeze(np.array(np.amax(X, 0)))[0])
#miny=float(np.squeeze(np.array(np.amin(X, 0)))[1])
#maxy=float(np.squeeze(np.array(np.amax(X, 0)))[1])
#plt.scatter(X1[:, 1], X1[:,2], marker='o', c=y)
#plt.axis([minx,maxx,miny,maxy])
#plt.show()
X = np.array(np.column_stack((np.matrix(X), np.ones(X.shape[0])))) #l'ultima colonna di X ha la costante per il bias
y[y==0]=-1

print(minimize_stochastic(cost_function, gradient_of_cost, X, y, np.zeros(nf+1)))
