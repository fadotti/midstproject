#test per il dataset
import pandas as pd
from src.svm import svm as SVM
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler 
import datetime
import random
import numpy as np
import time	 
import matplotlib.pyplot as plt
from src.perceptron import perceptron
scale = StandardScaler()

def F1(confusion_matrix):
    precision = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
    recall = confusion_matrix[1,1]/(confusion_matrix[0,1]+confusion_matrix[1,1])
    return 2 * precision * recall / (precision + recall)

df = pd.DataFrame(columns=['type','n_samples','n_test','time','accuracy','F1'])

for i in [1000,10000,100000]:
    (X,y) = make_blobs(n_samples = i, n_features = 2, centers = 2, cluster_std =1, random_state=0)
    (Xt,yt)= make_blobs(n_samples = int(i/2), n_features = 2, centers = 2, cluster_std =1, random_state=0)

    X = np.asmatrix(X)
    y[y==0]=-1
    X = scale.fit_transform(X)
    Xt = np.asmatrix(Xt)
    yt[yt==0]=-1
    Xt = scale.fit_transform(Xt)   
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    print(str(i)+' 1')
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df = df.append({'type':'standard','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 1')

    print(str(i)+' 2')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train() 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df = df.append({'type':'EBR','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 2')

    print(str(i)+' 3') 
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1,reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df = df.append({'type':'CDRM','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 3')
    
    print(str(i)+' 4')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df = df.append({'type':'CDRM+EBR','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 4')
    
    print(str(i)+' 5')
    p = perceptron(X,y,Xt,yt)
    time.sleep(1)
    starttime = datetime.datetime.now()
    p.train('seq') 
    endtime = datetime.datetime.now()
    conf,corr = p.test()
    df = df.append({'type':'perceptron','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 5')

(Xt,yt)= make_blobs(n_samples = int(i/2), n_features = 2, centers = 2, cluster_std =1, random_state=0)
plt.scatter(Xt[:, 0], Xt[:,1], marker='o', c=yt)
df.to_csv('output/separabili_ncresce.csv')
df.to_latex('output/separabili_ncresce.tex')

plt.savefig('output/separabili.png')

df2 = pd.DataFrame(columns=['type','n_samples','n_test','time','accuracy','F1'])


for i in [1000,10000,100000]:
    (X,y) = make_blobs(n_samples = i, n_features = 2, centers = 2, cluster_std =3, random_state=0)
    (Xt,yt)= make_blobs(n_samples = int(i/2), n_features = 2, centers = 2, cluster_std =3, random_state=0)

    X = np.asmatrix(X)
    y[y==0]=-1
    X = scale.fit_transform(X)
    Xt = np.asmatrix(Xt)
    yt[yt==0]=-1
    Xt = scale.fit_transform(Xt)   
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    print(str(i)+' 1')
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df2 = df2.append({'type':'standard','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 1')

    print(str(i)+' 2')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train() 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df2 = df2.append({'type':'EBR','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 2')

    print(str(i)+' 3') 
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1,reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df2 = df2.append({'type':'CDRM','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 3')
    
    print(str(i)+' 4')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df2 = df2.append({'type':'CDRM+EBR','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 4')

    print(str(i)+' 5')
    p = perceptron(X,y,Xt,yt)
    time.sleep(1)
    starttime = datetime.datetime.now()
    p.train('seq') 
    endtime = datetime.datetime.now()
    conf,corr = p.test()
    df2 = df2.append({'type':'perceptron','n_samples':i,'n_test':int(i/2),'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 5')


(Xt,yt)= make_blobs(n_samples = int(i/2), n_features = 2, centers = 2, cluster_std =3, random_state=0)
plt.scatter(Xt[:, 0], Xt[:,1], marker='o', c=yt)
df2.to_csv('output/nonseparabili_ncresce.csv')
df2.to_latex('output/nonseparabili_ncresce.tex')
plt.savefig('output/nonseparabili.png')



df3 = pd.DataFrame(columns=['type','n_samples','n_test','n_features','time','accuracy','F1'])


for i in [2,5,10,100]:
    (X,y) = make_blobs(n_samples = 10000, n_features = i, centers = 2, cluster_std =1, random_state=0)
    (Xt,yt)= make_blobs(n_samples = int(10000/2), n_features = i, centers = 2, cluster_std =1, random_state=0)

    X = np.asmatrix(X)
    y[y==0]=-1
    X = scale.fit_transform(X)
    Xt = np.asmatrix(Xt)
    yt[yt==0]=-1
    Xt = scale.fit_transform(Xt)   
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    print(str(i)+' 1')
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df3 = df3.append({'type':'standard','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 1')

    print(str(i)+' 2')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train() 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df3 = df3.append({'type':'EBR','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 2')

    print(str(i)+' 3') 
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1,reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df3 = df3.append({'type':'CDRM','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 3')
    
    print(str(i)+' 4')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df3 = df3.append({'type':'CDRM+EBR','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 4')

    print(str(i)+' 5')
    p = perceptron(X,y,Xt,yt)
    time.sleep(1)
    starttime = datetime.datetime.now()
    p.train('seq') 
    endtime = datetime.datetime.now()
    conf,corr = p.test()
    df3 = df3.append({'type':'perceptron','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 5')


df3.to_csv('output/separabili_pcresce.csv')
df3.to_latex('output/separabili_pcresce.tex')

df4 = pd.DataFrame(columns=['type','n_samples','n_test','n_features','time','accuracy','F1'])


for i in [2,5,10,100]:
    (X,y) = make_blobs(n_samples = 10000, n_features = i, centers = 2, cluster_std =3, random_state=0)
    (Xt,yt)= make_blobs(n_samples = int(10000/2), n_features = i, centers = 2, cluster_std =3, random_state=0)

    X = np.asmatrix(X)
    y[y==0]=-1
    X = scale.fit_transform(X)
    Xt = np.asmatrix(Xt)
    yt[yt==0]=-1
    Xt = scale.fit_transform(Xt)   
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    print(str(i)+' 1')
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df4 = df4.append({'type':'standard','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 1')

    print(str(i)+' 2')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train() 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df4 = df4.append({'type':'EBR','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 2')

    print(str(i)+' 3') 
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1,reduction=False) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df4 = df4.append({'type':'CDRM','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 3')
    
    print(str(i)+' 4')
    svm = SVM(X,y,Xt,yt,max_rounds=100)
    time.sleep(1)
    starttime = datetime.datetime.now()
    svm.train(0.1) 
    endtime = datetime.datetime.now()
    conf,corr = svm.test()
    df4 = df4.append({'type':'EBR+CDRM','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 4')

    print(str(i)+' 5')
    p = perceptron(X,y,Xt,yt)
    time.sleep(1)
    starttime = datetime.datetime.now()
    p.train('seq') 
    endtime = datetime.datetime.now()
    conf,corr = p.test()
    df4 = df4.append({'type':'perceptron','n_samples':10000,'n_test':int(10000/2),'n_features':i,'time':(starttime-endtime).total_seconds(),'accuracy':corr,'F1':F1(conf)}, ignore_index=True)
    print('end '+str(i)+' 5')


df4.to_csv('output/separabili_pcresce.csv')
df4.to_latex('output/separabili_pcresce.tex')
