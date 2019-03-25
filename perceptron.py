import numpy as np

class perceptron:



    def __init__(self, training_data, training_labels, test_data = None, test_labels = None, learn_rate = 0.0005, bias = 0.0, correctly_classified = 0.95, max_rounds = 100000):
        #aggiungo una colonna di -1 alla matrice di training per gestire il bias
        self.training_data = np.column_stack((np.matrix(training_data), np.negative(np.ones(training_data.shape[0]))))
        self.training_labels = np.array(training_labels)
        # pesi e bias inizializzati a 0. Ho un peso per ogni colonna della matrice di training
        #self.weights = np.random.rand(self.training_data.shape[1]) #pesi inizializzati random
        self.weights = np.zeros(self.training_data.shape[1])
        if test_data is not None:
            self.test_data = np.column_stack((np.matrix(test_data), np.negative(np.ones(test_data.shape[0]))))
        if test_labels is not None:
            self.test_labels = np.array(test_labels)
        self.learn_rate = learn_rate
        self.bias = bias
        self.max_rounds = max_rounds
        self.correctly_classified = correctly_classified



    def train(self, training_type = 'rand'):

        #training_type = "seq": controllo sequenziale del dataset
        #training_type = "rand": controllo casuale del dataset per un numero fissato di round


        # uso tutte le righe del training set sequenzialmente
        if training_type == "seq":

            dataset_rounds = 0 #numero di volte che ho ciclato sul dataset
            #numero di punti misclassificati
            previous_round_misclassified = self.training_data.shape[0] #assumo che all' inizio tutti siano misclassificati

            # ciclo il dataset finche:
            # non ho punti misclassificati oppure
            # il numero di punti misclassficati non cambia oppure
            # ho ciclato almeno 100 volte il dataset
            while(1):

                dataset_round_misclassified = 0
                current_row = 0

                for row in self.training_data:
                    #calcolo il valore di output con i pesi correnti
                    y = np.sign(np.dot(self.weights, np.squeeze(np.array(row))))
                    #controllo se l'etichetta ha lo stesso segno di y
                    if not y == self.training_labels[current_row]:
                        #se non sono uguali aggiorno i pesi spostando l'iperpiano verso x
                        #     vettore((p+1)x1)      +                scalare                                *     vettore((p+1)x1)
                        self.weights = self.weights + (self.learn_rate * self.training_labels[current_row]) * np.squeeze(np.array(row))
                        dataset_round_misclassified += 1
                    current_row += 1

                #finito di ciclare sul dataset controllo quanti sono stati misclassificati
                if dataset_round_misclassified == 0 or dataset_round_misclassified == previous_round_misclassified or dataset_rounds > 100:
                    return self.weights
                previous_round_misclassified = dataset_round_misclassified
                dataset_rounds += 1


        # uso solo alcune oss prese a caso nel training per un numero max_rounds di volte
        elif training_type == "rand":
            nrow = self.training_data.shape[0] #righe del training set
            for round in range(self.max_rounds):
                random_index = np.random.randint(nrow) #prendo un indice a caso
                row = self.training_data[random_index]
                #calcolo il valore di output con i pesi correnti
                y = np.sign(np.dot(self.weights, np.squeeze(np.array(row))))
                if not y == self.training_labels[random_index]:
                    #se non sono uguali aggiorno i pesi spostando l'iperpiano verso x
                    #     vettore((p+1)x1)      +                scalare                                 *     vettore((p+1)x1)
                    self.weights = self.weights + (self.learn_rate * self.training_labels[random_index]) * np.squeeze(np.array(row))

        return self.weights



    def test(self):
        confusion_matrix = np.matrix([[0, 0], [0, 0]])
        current_row = 0
        for row in self.test_data:
            #print(current_row)
            #previsione
            y = np.sign(np.dot(self.weights, np.squeeze(np.array(row))))
            #print(y)
            #aggiorno la matrice di confusione
            if(y == self.test_labels[current_row]):
                if (y > 0):
                    confusion_matrix[0,0] += 1
                else:
                    confusion_matrix[1,1] += 1
            else:
                if (y - self.test_labels[current_row]) == 2: # 1 - (-1) = 2
                    confusion_matrix[0,1] += 1
                else:
                    confusion_matrix[1,0] += 1
            current_row += 1

        correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/confusion_matrix.sum())*100
        return confusion_matrix, correct_classification
