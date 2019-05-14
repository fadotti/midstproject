#usa questo comando per eseguire da terminale questo file
#python mr_perceptron.py --l_rate 0.01 --current_weights weights.txt data1.txt

from mrjob.job import MRJob
import numpy as np
from src.svm import svm
import os
#i pesi sono salvati su un file nel formato 
#i\tw_i\n
#i\tw_i\n
#...
#è necessario avere un file con i pesi di partenza altrimenti non esegue
#con questa funzione li tiro fuori dal file

def get_globalv(file):
    vectors = np.loadtxt(file,dtype=float)
    return vectors

class mr_svm(MRJob):

    #configure_args è per aggiungere parametri quando chiamo l'esecuzione da terminale
    def configure_args(self):
        super(mr_svm, self).configure_args()
        self.add_passthru_arg('--l_rate', default=0.05) #learning rate
        self.add_file_arg('--global_vectors', default= 'globalt.txt')#i pesi dell'iterazione precendente sono stati salvati in un file

    #mapper_init viene eseguito sempre prima del mapper
    def mapper_init(self):
        self.l_rate = float(self.options.l_rate) #prendo i learning rate
        self.global_vectors = get_globalv(self.options.global_vectors)
    #uso mapper_raw se voglio tutto il file e non solo la riga singola di file
    #dentro mapper_raw decido come leggere il file col solito f = open(file)
    #https://mrjob.readthedocs.io/en/latest/guides/writing-mrjobs.html#passing-entire-files-to-the-mapper
    
    #per quanto riguarda questo mapper faccio quello che è scritto nel libro
    def mapper_raw(self, path, uri):
        data = np.loadtxt(path)
        chunk = os.path.split(path)[-1]
        if(self.global_vectors.size!=0):
            data = np.unique(np.append(data,self.global_vectors,axis=0),axis=0)
        yield chunk,data.tolist()
            
    def reducer(self, key, values):
        vv = np.squeeze(np.array(list(values)))
        X = vv[:,:-1]  
        y = vv[:,-1]
        r_svm = svm(X,y,max_rounds=100)
        r_svm.train()
        r_svm.get_sv()
        vectors = list(r_svm.support_vectors)
        a = np.column_stack((X[vectors,:],y[vectors]))
        yield key,a.tolist()

if __name__ == '__main__':
    mr_svm.run()