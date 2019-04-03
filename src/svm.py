import numpy as np
import random as rn
import math 
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
                 hinge_function = lambda w,x,yi,C:  0 if (yi * np.dot(x,w)) >= 1 else  - C * (yi * x)
                 ):
        self.training_data = np.column_stack((np.matrix(training_data),np.negative(np.ones(training_data.shape[0]))))
        self.training_labels = np.array(training_labels)
        self.weights = np.zeros(self.training_data.shape[1])
        if test_data is not None:
            self.test_data = np.column_stack((np.matrix(test_data), np.negative(np.ones(test_data.shape[0]))))
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
        

    # STATICS 
    def __select_C(self):
        return 1

    
    def __stochastic_gradient_descent(self):
        w = np.zeros(self.training_data.shape[1])
        min_w, min_f = None, float('inf') 
        l_rate = self.learn_rate
        no_improv = 0 
        x = list(range(self.training_data.shape[0]))

        while no_improv < 100:
            f_value = self.cost_function(w,self.training_data, self.training_labels,self.C)
            print(f_value)
            if f_value < min_f: 
                min_w, min_f = w, f_value 
                no_improv = 0
            else: 
                no_improv += 1
                l_rate *= 0.9
            rn.shuffle(x)

            for i in x:
                hinge_i = self.hinge_function(w,np.squeeze(np.array(self.training_data[i])), self.training_labels[i], self.C)
                w = w - l_rate * (w+hinge_i)
            
        return min_w, min_f
    
    def __batch_gradient_descent(self,batch_size):
        w = np.zeros(self.training_data.shape[1])
        hinge_i= w
        min_w, min_f = None, float('inf') 
        l_rate = self.learn_rate 
        x = list(range(self.training_data.shape[0]))
        for i in range(self.max_rounds):
            f_value = self.cost_function(w,self.training_data, self.training_labels,self.C)
            print(f_value)
            #input("Press Enter to continue...")

            if f_value < min_f: 
                min_w, min_f = w, f_value 
                
            for k in range(math.ceil(len(x)/batch_size)):
                for j in range(batch_size):
                    index = j+k*(batch_size) 
                    if(index < len(x)):
                        hinge_i = hinge_i+self.hinge_function(w,np.squeeze(np.array(self.training_data[index])), self.training_labels[index], self.C)    
                
                w = w - l_rate * (w+hinge_i)
            
        return min_w, min_f 

    # CLASS METHODS

    def predict(self,row):
        return np.sign(np.dot(self.weights, np.squeeze(np.array(row)))).astype('int')

    
    
        
    def train(self,method='stochastic',batch_size = None):
        if batch_size==None:
            batch_size=len(self.training_labels)
        if method=='stochastic':
            self.weights,min_f = self.__stochastic_gradient_descent()
        elif method =='batch':
            self.weights,min_f = self.__batch_gradient_descent(batch_size)
        return self.weights
    
    def test(self):
        confusion_matrix = np.matrix([[0, 0], [0, 0]])
        current_row = 0
        for row in self.test_data:
            y = self.predict(row)
            confusion_matrix[int((self.test_labels[current_row]+1)/2),int((y+1)/2)]=confusion_matrix[int((self.test_labels[current_row]+1)/2),int((y+1)/2)]+1

            current_row += 1

        correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
        return confusion_matrix, correct_classification

    
    
    

    

