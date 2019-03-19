import numpy as np

class perceptron:
    """Class defining perceptrons"""
    @staticmethod
    def update_weights(self,X,y):
        return self.weights+self.learn_rate*
    
    def __init__(self,learn_rate=0.05,
                 training_data=None,
                 training_labels=None,
                 test_data=None,
                 test_labels=None):
        self.learn_rate = learn_rate
        if training_data is not None:
            self.training_data = np.matrix(training_data)
        if training_labels is not None:
            self.training_labels = np.array(training_labels)
        if test_data is not None:
            self.test_data = np.matrix(test_data)
        if test_labels is not None:
            self.test_labels = np.array(test_labels)
        self.weights = np.zeros(self.training_data.shape[2])
        