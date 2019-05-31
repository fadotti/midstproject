#ho usato due dataset: 
# data1.txt con 2 features e richiede quindi il file weights iniziale con 2 pesi inizializzati
# sparse.txt con 10 features dicotomiche generate da die binomiali diverse e richiede il file weights iniziale con 10 pesi inizializzati
# con sparse.txt la classificazione fa un po schifo probabilemnte perchè ho generato male i dati
# in nessuno dei due casi ho tenuto conto del bias ma per quello basta aggiungere una colonna di -1 alla matrice X e aggiungere un peso al file weights


from mr_perceptron import mr_train
import shutil as sh
import numpy as np

#lunghezza in righe del file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

#conta il numero di missclassificati per fermare l'algoritmo
def count_missclassified(data, weights):

    #tecnicamente non mi serve ma la calcolo per vedere la percentuale di corretta classificazione ad ogni iterazione
    confusion_matrix = np.matrix([[0, 0], [0, 0]])
    missclassified = 0

    #tiro fuori i pesi
    f = open(weights, 'r')
    weights = []
    for string in f.readlines():
        string = string[1:]
        string = string.rstrip('\t')
        weights.append(float(string))
    weights = np.array(weights)
    f.close()

    #tiro fuori i dati
    f = open(data, "r")
    for row in f:
        values = np.array([float(i) for i in row.split()])
        label = values[-1]
        values = np.delete(values, -1)
        #vedo se sono misclassificati e aggiorno la matrice di confusione
        y =  np.sign(np.dot(weights, values))
        if not y == label:
            missclassified += 1
        confusion_matrix[int((label+1)/2),int((y+1)/2)] = confusion_matrix[int((label+1)/2),int((y+1)/2)]+1
    
    correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
    print(correct_classification)#per vedere a ogni iterazione come varia la corretta classificazione
    return missclassified


#assumo che all' inizio tutti siano misclassificati
previous_round_misclassified = file_len('data1.txt') 
dataset_rounds = 0

while True:
    dataset_round_misclassified = 0
    #con questa riga è come se chiamassi mr_perceptron da terminale con i parametri definiti da args
    mr_job = mr_train(args=['--l_rate', '0.01', '--current_weights', 'weights.txt','data1.txt'])
    #salvo l'output dell' iterazione del dataset in un file temporaneo weights1.txt
    #perchè a quanto pare non posso lavorare sullo stesso file dei pesi
    #(probabilmente perchè quando eseguo mr_perceptron apre lo stesso file)
    f = open('weights1.txt', 'w')
    with mr_job.make_runner() as runner:
        runner.run()#run mr_perceptron
        #prendo l'output che dovrei ottenere e lo metto dentro il file temporaneo
        #nel formato
        #i\tw_i\n
        for key, value in mr_job.parse_output(runner.cat_output()):
            f.write(str(key) +'\t'+ str(value)+'\n')
    f.close()
    #i nuovi pesi diventano quelli appena calcolati nell' iterazione del dataset
    sh.copyfile('weights1.txt', 'weights.txt')#devo quindi copiare il file temporaneo sul file dei pesi

    #conto quanti sono misclassificati dopo l'aggiornamento dei pesi
    dataset_round_misclassified = count_missclassified('data1.txt', 'weights.txt')
    #solito criterio di stop
    if dataset_round_misclassified == 0 or dataset_round_misclassified == previous_round_misclassified or dataset_rounds > 100:
        break
    previous_round_misclassified = dataset_round_misclassified
    dataset_rounds += 1


