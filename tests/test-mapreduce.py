import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm as _svm 
from sklearn.preprocessing import StandardScaler 
import datetime
import random
import time	 

from mapreduce import jobbolino
import io
(X,y) = make_blobs(n_samples = 10000, n_features = 2, centers = 2, cluster_std =1, random_state=99)
y[y==0]=-1
scale = StandardScaler()

X = scale.fit_transform(X)
out = np.column_stack((list(range(len(y))),X,y))
np.savetxt('data.txt',out,delimiter=' ',fmt='%i %f %f %i')
mr_job = jobbolino(args=['--no-conf'])
mr_job.sandbox(stdin=open('data.txt','rb'))
values = []
with mr_job.make_runner() as runner:
    runner.run()
    for line in runner.stream_output():
        key, value = mr_job.parse_output_line(line)
        values.extend(value)
values = list(map(int,values))
p = svm(X[values],y[values])
p.train()
p2 = svm(X,y)
p2.train(0.1)
cose = _svm.SVC(kernel='linear')
k = cose.fit(X,y)
xx = np.linspace(-2.5, 2.5)
w = p.weights
#mapreduce
a =  -w[0]/w[1]
yy = a*xx
#per scikitsvm
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx
#svm normale
w2 = p2.weights
a2 =  -w2[0]/w2[1]
yy3 = a2*xx

plt.scatter(X[:, 0], X[:,1], marker='o', c=y)
plt.plot(xx,yy,color='blue')#mapreduce
plt.plot(xx,yy1,color='orange')#scikit
plt.plot(xx,yy3,color='red')#svm
plt.scatter(X[values, 0], X[values,1], marker='o', color='green')
plt.scatter(k.support_vectors_[:,0],k.support_vectors_[:,1])
plt.scatter(X[list(p.support_vectors),0],X[list(p.support_vectors),1],color='pink')
plt.scatter(X[list(p2.support_vectors),0],X[list(p2.support_vectors),1],color='violet')
plt.show()