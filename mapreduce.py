
from mrjob.job import MRJob
import math
from src.svm import svm
import numpy as np
from sklearn.preprocessing import StandardScaler 

class jobbolino(MRJob):
    def mapper(self,_,line ):
        line = line.rstrip().split(' ')
        line = [float(x) for x in line]
        key = math.floor(line[0]/1000)
        yield key,line  

    def reducer (self,key,values ):
        scale = StandardScaler()
        vals = np.matrix(list(values))
        i = np.squeeze(np.array(vals[:,0]))
        X = np.squeeze(vals[:,1:3])
        y = np.squeeze(np.array(vals[:,3]))
        p = svm(X,y)
        p.train(0.5)
        p.get_sv()
        yield key,list(i[np.array(list(p.support_vectors))])

if __name__ == '__main__':
    jobbolino.run()



