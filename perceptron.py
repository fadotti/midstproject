import numpy as np

class perceptron:
    """Class defining perceptrons"""
    
    def update_weights(self):
        """TODO: fix, parallelizzare"""
        self.weights=np.array(self.weights+self.learn_rate*np.dot(self.training_labels,self.training_data))[0]
    
    def __init__(self,learn_rate=10,
                 training_data=None,
                 training_labels=None,
                 test_data=None,
                 test_labels=None,
                 bias = 0.0,
                 correctly_classified = 0.95,
                 max_rounds = 10000000, 
                 chunk_size=False):

        self.learn_rate = learn_rate
        if training_data is not None:
            self.training_data = np.matrix(training_data)
        if training_labels is not None:
            self.training_labels = np.array(training_labels)
        if test_data is not None:
            self.test_data = np.matrix(test_data)
        if test_labels is not None:
            self.test_labels = np.array(test_labels)
        self.weights = np.zeros(self.training_data.shape[1])
        self.chunk_size = chunk_size
        self.bias = bias
        self.max_rounds=max_rounds
        self.correctly_classified=correctly_classified
        
    def train(self):
        y = np.zeros(self.training_data.shape[0])
        current_round = 0
        while not sum((self.training_labels-y)==0)/len(self.training_labels)>self.correctly_classified and current_round < self.max_rounds:
            current_round = current_round + 1
            y = self.predict(self.training_data)
            print 'y=',y
            print 'labels=',self.training_labels
            if all(self.training_labels==y):
                return
            self.update_weights()
    
    def test(self):
        y = predict(self.test_data)
        confusion_matrix = np.matrix([[0,0],[0,0]])
        for i in xrange(len(y)):
            confusion_matrix[(self.test_labels[i]+1)/2,(y[i]+1)/2]=confusion_matrix[(self.test_labels[i]+1)/2,(y[i]+1)/2]+1
        return confusion_matrix
    
    def predict(self,X):
        return np.array(np.sign(np.dot(X,self.weights)+self.bias))[0]
    
    