#usa questo comando per eseguire da terminale questo file
#python mr_perceptron.py --l_rate 0.01 --current_weights weights.txt data1.txt

from mrjob.job import MRJob
import numpy as np

#i pesi sono salvati su un file nel formato 
#i\tw_i\n
#i\tw_i\n
#...
#è necessario avere un file con i pesi di partenza altrimenti non esegue
#con questa funzione li tiro fuori dal file
def get_w(file):
    f = open(file, 'r')
    w = []
    for string in f.readlines():
        string = string[1:]
        string= string.rstrip('\t')
        w.append(float(string))
    w = np.array(w)
    f.close()
    return w

class mr_train(MRJob):

    #configure_args è per aggiungere parametri quando chiamo l'esecuzione da terminale
    def configure_args(self):
        super(mr_train, self).configure_args()
        self.add_passthru_arg('--l_rate', default=0.05) #learning rate
        self.add_file_arg('--current_weights', default= 'weights.txt')#i pesi dell'iterazione precendente sono stati salvati in un file

    #mapper_init viene eseguito sempre prima del mapper
    def mapper_init(self):
        self.l_rate = float(self.options.l_rate) #prendo i learning rate
        self.w = get_w(self.options.current_weights) #e mi tiro fuori i pesi dal file

    #uso mapper_raw se voglio tutto il file e non solo la riga singola di file
    #dentro mapper_raw decido come leggere il file col solito f = open(file)
    #https://mrjob.readthedocs.io/en/latest/guides/writing-mrjobs.html#passing-entire-files-to-the-mapper
    
    #per quanto riguarda questo mapper faccio quello che è scritto nel libro
    def mapper(self, _, line):
        row = [float(i) for i in line.split()]
        label = row[-1]
        x = np.array(row[:-1])
        y = np.sign(np.dot(x,self.w))
        if not y == label:
            for i, xi in enumerate(x):
                if not xi == 0:
                    yield i, self.l_rate*label*xi

    #reducer_init viene eseguito prima di ogni reducer
    def reducer_init(self):
        self.w = get_w(self.options.current_weights)# mi tiro fuori i pesi per poi poterli aggiornasre
        #non sono riuscito a farglielo fare solo una volta quindi devo farlo sia prima del mapper che del reducer
        #probabilmente dovrei sovrascrivere l'init della classe ma poi non riesco a passargli parametri extra e boh

    def reducer(self, key, values):
        s = sum(values) #somma di tutte le i-esime componenti per ogni riga
        yield key, self.w[key] + s #aggiorno i pesi

if __name__ == '__main__':
    mr_train.run()