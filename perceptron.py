import numpy as np

class perceptron:
    """Class defining perceptrons"""
    
    def update_weights(self,y):
        """TODO: fix, parallelizzare"""
        self.weights=np.array(self.weights+self.learn_rate*np.dot(np.transpose(self.training_data),(self.training_labels-y)))[0]
    
    def __init__(self,learn_rate=0.0005,
                 training_data=None,
                 training_labels=None,
                 test_data=None,
                 test_labels=None,
                 bias = 0.0,
                 correctly_classified = 0.95,
                 max_rounds = 10000, 
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
        self.weights = np.random.rand(self.training_data.shape[1])
        self.chunk_size = chunk_size
        self.bias = bias
        self.max_rounds=max_rounds
        self.correctly_classified=correctly_classified
        
    def train(self):
        y = np.zeros(self.training_data.shape[0])
        current_round = 0
        print('Training')
        self.printProgressBar(0, self.max_rounds, prefix = 'Progress:', suffix = 'Rounds', length = 50)
        while not sum((self.training_labels-y)==0)/len(self.training_labels)>self.correctly_classified and current_round < self.max_rounds:
            current_round = current_round + 1
            self.printProgressBar(current_round, self.max_rounds, prefix = 'Progress:', suffix = 'Rounds', length = 50)
            
            y = self.predict(self.training_data)
            
            self.update_weights(y)
        confusion_matrix = np.matrix([[0,0],[0,0]])       
         
        for i in range(len(y)):
            confusion_matrix[int((self.training_labels[i]+1)/2),int((y[i]+1)/2)]=confusion_matrix[int((self.training_labels[i]+1)/2),int((y[i]+1)/2)]+1
        print('\nConfusion matrix')
        print(confusion_matrix)
        print('Correct classification')
        print((confusion_matrix[0,0]+confusion_matrix[1,1])/confusion_matrix.sum())
    
    def test(self):
        y = self.predict(self.test_data)
        confusion_matrix = np.matrix([[0,0],[0,0]])
        for i in range(len(y)):
            confusion_matrix[int((self.test_labels[i]+1)/2),int((y[i]+1)/2)]=confusion_matrix[int((self.test_labels[i]+1)/2),int((y[i]+1)/2)]+1
        return confusion_matrix
    
    def predict(self,X):
        return np.array(np.sign(np.dot(X,self.weights)+self.bias))[0]
    
    def printProgressBar (self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        if iteration == total: 
            print()