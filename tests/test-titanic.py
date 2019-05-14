import pandas as pd
import numpy as np
import patsy 
from src.svm import svm
from sklearn import svm as _svm  

train = pd.read_csv('datasets/titanic/train.csv',sep=',',header=0)


train = train.drop([61,829],axis=0)
y,X2 = patsy.dmatrices('Survived~1+Pclass+Sex+SibSp+Parch+Fare+Embarked',train)

y = np.squeeze(np.array(y))
X2 = np.matrix(X2)
y[y==0]=-1

test = pd.read_csv('datasets/titanic/test.csv',sep=',',header=0)
testsurv = pd.read_csv('datasets/titanic/gender_submission.csv',sep=',',header=0)

test.isna().sum()
test.loc[test.loc[:,'Fare'].isna(),'Fare']
test = test.drop([152],axis=0)
testsurv = testsurv.drop([152],axis=0)
test_complete = pd.merge(test,testsurv,on='PassengerId')
yt,X2t = patsy.dmatrices('Survived~1+Pclass+Sex+SibSp+Parch+Fare+Embarked',test_complete)
yt = np.squeeze(np.array(yt))
X2t = np.matrix(X2t)
yt[yt==0]=-1


p = svm(X2,y,X2t,yt,max_rounds=10000)
p.train(reduction=False)
p.test()

cose = _svm.SVC(kernel='linear')
k = cose.fit(X2,y)

y_pred = k.predict(X2t)
confusion_matrix = np.matrix([[0, 0], [0, 0]])
current_row = 0
for i in range(len(y_pred)):
    confusion_matrix[int((yt[i]+1)/2),int((y_pred[i]+1)/2)]=confusion_matrix[int((yt[i]+1)/2),int((y_pred[i]+1)/2)]+1
    current_row += 1
correct_classification = ((confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix.sum()))*100
confusion_matrix
correct_classification