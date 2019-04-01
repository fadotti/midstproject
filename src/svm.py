import numpy as np
import random as rn

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
                 cost_function = lambda w,X,y,C:(1/2) * np.sum(w**2) + C * np.sum(map(lambda xi,yi: max(0, 1 - yi * np.dot(xi,w)),X, y)),
                 gradient_function = lambda w,x,yi,C:  w if (yi * np.dot(w,x)) >= 1 else w - C * (yi * x)
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
        self.C = 42
        self.cost_function = cost_function
        self.gradient_function = gradient_function
        
    
    def predict(self,row):
        return np.sign(np.dot(self.weights, np.squeeze(np.array(row)))).astype('int')
        
    def __update_weights(self,label,pattern):
        self.weights =np.squeeze(np.array(self.weights - self.learn_rate*(self.weights - self.C * label * pattern)))
        self.weights = self.weights/np.sqrt(sum(self.weights**2))

    def stochastic_gradient_descent(self):

        w = np.zeros(self.training_data.shape[1])
        min_w, min_f = None, float('inf') 
        l_rate = self.learn_rate
        no_improv = 0 
        x = list(range(self.training_data.shape[0]))

        while no_improv < 100:
            f_value = self.cost_function(w,self.training_data, self.training_labels,self.C)
            if f_value < min_f: 
                min_w, min_f = w, f_value 
                no_improv = 0
            else: 
                no_improv += 1
                l_rate *= 0.9
            rn.shuffle(x)

            for i in x:
                gradient_i = self.gradient_function(w,self.training_data[i,:], self.training_labels[i], self.C)
                w = w - l_rate * gradient_i 

        return min_w, min_f
        
    def train(self):
        
        for i in xrange(self.max_rounds):
            n_row = np.random.randint(len(self.training_labels))
            row = self.training_data[n_row]
            label = self.training_labels[n_row]
            y = self.predict(row)
            if label*y < 1:
                self.__update_weights(label,row)
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


    
    

    

