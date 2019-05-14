import numpy as np
import random as rn
import math 
import multiprocessing
import matplotlib.pyplot as plt
import datetime

def single_cdrm(args):
    i=args[0]
    m1=args[1]
    m2 = args[2]
    x = args[3]
    y = args[4]
    treshold = args[5]
    d1 = np.sqrt(np.sum(np.squeeze(np.array((x-m1)))**2)).astype('float')
    d2 = np.sqrt(np.sum(np.squeeze(np.array((x-m2)))**2)).astype('float')
    
    if((d1/d2)**y > treshold):
        return i




class svm:

    def __init__(self, 
                 training_data, 
                 training_labels, 
                 test_data = None, 
                 test_labels = None, 
                 learn_rate = 0.0005, 
                 bias = 0.0, 
                 correctly_classified = 0.95, 
                 max_rounds = 100000,
                 kernel = None,
                 C = None,
                 cost_function = lambda w,X,y,C:(1/2) * np.sum(w**2) + C * np.sum(list(map(lambda xi,yi: max(0, 1 - yi * np.dot(xi,w)),X, y))),
                 hinge_function = lambda w,x,yi,C:  0 if (yi * np.dot(x,w)) >= 1  else  - C * (yi * x),
                 cores = multiprocessing.cpu_count()
                 ):
        self.training_data = np.column_stack((np.matrix(training_data),np.ones(training_data.shape[0])))
        self.training_labels = np.array(training_labels)
        self.weights = np.zeros(self.training_data.shape[1])
        if test_data is not None:
            self.test_data = np.column_stack((np.matrix(test_data),np.ones(test_data.shape[0])))
        if test_labels is not None:
            self.test_labels = np.array(test_labels)
        self.learn_rate = learn_rate
        self.bias = bias
        self.max_rounds = max_rounds
        self.correctly_classified = correctly_classified
        self.kernel = kernel
        self.C = C if C is not None else self.__select_C()
        self.cost_function = cost_function
        self.hinge_function = hinge_function
        self.treshold = 0.001
        self.support_vectors = set()
        self.cores = cores
    # STATICS 
    def __select_C(self):
        return 1
    


    def __CDRM(self,treshold):
        if(treshold==float('inf')):
            return list(range(len(self.training_labels)))
        m1 = np.mean(self.training_data[self.training_labels==1],0)
        m2 = np.mean(self.training_data[self.training_labels==-1],0)
        vectors = []
        pool = multiprocessing.Pool(self.cores)
        vectors = pool.map(single_cdrm,list(
            map(lambda i,m1,m2,x,y,treshold:[i,m1,m2,x,y,treshold] ,
            list(
                range(len(self.training_labels))),
                [m1]*len(self.training_labels),
                [m2]*len(self.training_labels),
                self.training_data,
                self.training_labels,
                [treshold]*len(self.training_labels)
                )
            )
            )
        """ for i in range(len(self.training_labels)):
            d1 = np.sqrt(np.sum(np.squeeze(np.array((self.training_data[i]-m1)))**2)).astype('float')
            d2 = np.sqrt(np.sum(np.squeeze(np.array((self.training_data[i]-m2)))**2)).astype('float')
            
            if((d1/d2)**self.training_labels[i] > treshold):
               vectors.append(i) """
        return [x for x in vectors if x is not None]
    
    
    # CLASS METHODS

    def predict(self,row):
        return np.sign(np.dot(self.weights, np.squeeze(np.array(row)))).astype('int')

    
    def train(self,treshold=float('inf'),plots=False,reduction = True):
        w = self.weights
        n = len(self.training_labels)
        min_w, min_f = None, float('inf') 
        l_rate = self.learn_rate
        no_improv = 0
        epoch = 0
        j=0
        x = self.__CDRM(treshold)
        tresh=1
        while no_improv < 100 and epoch<self.max_rounds :
            
            f_value = self.cost_function(w,self.training_data[x], self.training_labels[x],self.C)
            if  min_f-f_value>self.treshold: 
                min_w, min_f = w, f_value 
                no_improv = 0
                epoch=epoch+1
            else:
                epoch = 0
                no_improv += 1
                l_rate *= 0.9
            rn.shuffle(x)

            for i in x:
                hinge_i = self.hinge_function(w,np.squeeze(np.array(self.training_data[i])), self.training_labels[i], self.C)
                w = w - l_rate * (w/len(x)+hinge_i)
            if(plots):
                self.__plot_to_file(j,w)
                j=j+1
            if(reduction):
                self.get_sv(w,x,tresh)
                tresh=tresh*0.9
                x = list(self.support_vectors)
            
        self.weights = min_w    
        return self.weights

    def __plot_to_file(self,i,w):
        xx = np.linspace(-2.5, 2.5)
        a =  -w[0]/w[1]
        yy = a*xx-w[2]/w[1]
        plt.scatter(np.squeeze(np.array(self.training_data))[:, 0], np.squeeze(np.array(self.training_data))[:,1], marker='o',c=np.array(np.squeeze(self.training_labels)))
        plt.scatter(np.squeeze(np.array(self.training_data))[(np.array(list(self.support_vectors))).astype(int), 0], np.squeeze(np.array(self.training_data))[(np.array(list(self.support_vectors))).astype(int),1], marker='*', color='green')
        plt.plot(xx,yy,color='blue')
        axes = plt.gca()
        axes.set_ylim([np.min(np.squeeze(np.array(self.training_data))[:, 1]).astype('float'),np.max(np.squeeze(np.array(self.training_data))[:, 1]).astype('float')])
        plt.savefig('plots/'+str(i)+'.png') 
        
        plt.clf()    


    def test(self):

        confusion_matrix = np.matrix([[0, 0], [0, 0]])
        current_row = 0
        for row in self.test_data:
            y = self.predict(row)
            confusion_matrix[int((self.test_labels[current_row]+1)/2),int((y+1)/2)]=confusion_matrix[int((self.test_labels[current_row]+1)/2),int((y+1)/2)]+1
            current_row += 1
        correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
        return confusion_matrix, correct_classification

    def get_sv(self,w=None,X=None,treshold=0.01):
            self.support_vectors = set()
            n = X if X is not None else list(range(self.training_data.shape[0]))
            if w is None: 
                w=self.weights
            for j in n:
                if -1-treshold<=np.dot(np.squeeze(np.array(self.training_data[j])),w) < 1+treshold or np.sign(np.dot(w, np.squeeze(np.array(self.training_data[j])))).astype('int')!=self.training_labels[j]:
                    self.support_vectors.add(j)
 

    

