import numpy as np


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
                 kernel = None
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
        
    
    def predict(self,row):
        return np.sign(np.dot(self.weights, np.squeeze(np.array(row)))).astype('int')
        
    def __update_weights(self,label,pattern):
        self.weights =np.squeeze(np.array(self.weights - self.learn_rate*(self.weights - self.C * label * pattern)))
        self.weights = self.weights/np.sqrt(sum(self.weights**2))

    def __stochastic_gradient_descent(self):
        print('WIP')
        
    def train(self, training_type = 'rand'):
        
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
            #print(current_row)
            #previsione
            y = self.predict(row)
            #print(y)
            #aggiorno la matrice di confusione
            confusion_matrix[int((self.test_labels[current_row]+1)/2),int((y+1)/2)]=confusion_matrix[int((self.test_labels[current_row]+1)/2),int((y+1)/2)]+1

            current_row += 1

        correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
        return confusion_matrix, correct_classification


    
    

    

