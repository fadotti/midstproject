import numpy as np
import matplotlib.pyplot as plt
from src.svm import svm
from sklearn import svm as _svm  
from sklearn.preprocessing import StandardScaler 
scale = StandardScaler()


train = np.genfromtxt('./datasets/train_data_generated.csv',delimiter=',',skip_header=1)
print({i:len(train[train[:,-1]==i,:]) for i in set(train[:,-1])})

y=train[:,-1]
X=train[:,:-1]

test = np.genfromtxt('./datasets/test_data_generated.csv',delimiter=',',skip_header=1)
print({i:len(test[test[:,-1]==i,:]) for i in set(test[:,-1])})

yt=test[:,-1]
Xt=test[:,:-1]

X = scale.fit_transform(X)
Xt = scale.fit_transform(Xt)
mysvm = svm(X,y,Xt,yt,max_rounds=500)
#coeff = mysvm.train(0.1,plots=True)
coeff = mysvm.train(0.1)
mysvm.test()


cose = _svm.SVC(kernel='linear',)
k = cose.fit(X,y)
np.sum((mysvm.weights[:-1]-k.coef_)**2)

y_pred = k.predict(Xt)
confusion_matrix = np.matrix([[0, 0], [0, 0]])
current_row = 0
for i in range(len(y_pred)):
    confusion_matrix[int((yt[i]+1)/2),int((y_pred[i]+1)/2)]=confusion_matrix[int((yt[i]+1)/2),int((y_pred[i]+1)/2)]+1
    current_row += 1
correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
confusion_matrix
correct_classification

xx = np.linspace(-2.5, 2.5)
w=mysvm.weights
#per scikitsvm
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx+k.intercept_[0]
a =  -w[0]/w[1]
#a2 =  -w2[0]/w2[1]
yy = a*xx-w[2]/w[1]
#yy2 = a2*xx

plt.scatter(Xt[yt==1, 0], Xt[yt==1,1], marker='o', color='green')
plt.scatter(Xt[yt==-1, 0], Xt[yt==-1,1], marker='*', color='purple')

plt.plot(xx,yy,color='blue')
plt.plot(xx,yy1,color='orange')
plt.scatter(k.support_vectors_[:,0], k.support_vectors_[:,1], marker='*', color='yellow')
mysvm.get_sv()
plt.scatter(X[np.array(list(mysvm.support_vectors)), 0], X[np.array(list(mysvm.support_vectors)),1], marker='+', color='dodgerblue')
# axes = plt.gca()
# axes.set_ylim([-5,5])
plt.show()