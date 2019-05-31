#ho usato due dataset: 
# data1.txt con 2 features e richiede quindi il file weights iniziale con 2 pesi inizializzati
# sparse.txt con 10 features dicotomiche generate da die binomiali diverse e richiede il file weights iniziale con 10 pesi inizializzati
# con sparse.txt la classificazione fa un po schifo probabilemnte perchè ho generato male i dati
# in nessuno dei due casi ho tenuto conto del bias ma per quello basta aggiungere una colonna di -1 alla matrice X e aggiungere un peso al file weights


from mr_svm import mr_svm
import shutil as sh
import numpy as np
import os 
import sys
import copy 
folder = 'datasets/data/'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(folder):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
args = ['--l_rate', '0.01', '--global_vectors', 'global_vectors.txt']
args.extend(files)

f = open('global_vectors.txt','w')
f.close()


while True:
    mr_job = mr_svm(args=args)
    global_vectors = np.loadtxt('global_vectors.txt',dtype=float)
    gv = np.array([])
    with mr_job.make_runner() as runner:
        runner.run()#run mr_perceptron
        #prendo l'output che dovrei ottenere e lo metto dentro il file temporaneo
        #nel formato
        #i\tw_i\n
        for key, value in mr_job.parse_output(runner.cat_output()):
            for line in value:
                gv = np.append(gv,line)    
    gv = gv.reshape(int(len(gv)/len(line)),len(line))  
    gv = np.unique(gv,axis=0)     
    if(np.array_equal(np.sort(gv,axis=0), np.sort(global_vectors,axis=0))):
        break
    np.savetxt('global_vectors.txt',gv,fmt='%s')


from src.svm import svm
import matplotlib.pyplot as plt
from sklearn import svm as _svm  

total_data = np.loadtxt('mr_perceptron/data1.txt')
XX = total_data[:,:(-1)]
yy = total_data[:,-1]
data = np.loadtxt('global_vectors.txt')
X = data[:,:(-1)]
y = data[:,-1]
p = svm(X,y,XX,yy)
w = p.train(0.1)
p.test()
cose = _svm.SVC(kernel='linear')
k = cose.fit(X,y)
np.sum((p.weights[:-1]-k.coef_)**2)

y_pred = k.predict(XX)
confusion_matrix = np.matrix([[0, 0], [0, 0]])
current_row = 0
for i in range(len(y_pred)):
    confusion_matrix[int((yy[i]+1)/2),int((y_pred[i]+1)/2)]=confusion_matrix[int((yy[i]+1)/2),int((y_pred[i]+1)/2)]+1
    current_row += 1
correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
confusion_matrix
correct_classification


plt.scatter(XX[:, 0], XX[:,1], marker='o', c=yy)
xx = np.linspace(-5, 5)
a =  -w[0]/w[1]
yy2 = a*xx-w[2]/w[1]
b =  -k.coef_[0][0]/k.coef_[0][1]
yy1 = b*xx-k.intercept_[0]/k.coef_[0][1]
plt.scatter(X[np.array(list(p.support_vectors)), 0], X[np.array(list(p.support_vectors)),1], marker='+', color='dodgerblue')
plt.scatter(k.support_vectors_[:,0], k.support_vectors_[:,1], marker='*', color='red')

plt.plot(xx,yy2,color='blue')
plt.plot(xx,yy1,color='red')
plt.show()
