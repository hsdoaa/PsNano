# To set seed random number in order to reproducable results in keras
from numpy.random import seed
seed(4)
import tensorflow
tensorflow.random.set_seed(1234)
########################################
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd
from statistics import mean, median
import numpy as np
import sys
import pseudoExtractor as ps
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from numpy import *
import signalExtractor as signal
import plot_learning_curves as plc
from sklearn.preprocessing import MinMaxScaler #For feature scaling
scaler = MinMaxScaler()

classifier =svm.SVC(gamma='scale',C=2,probability=True)

#get signal length:
def get_signal_length(x):  
    return len(x)

#start pseudoExtractor 
controlHela, pseudoHela = ps.get_Hela()
#omit file name
drp = [0, 2]
controlHela = controlHela.drop(drp, axis=1) 
pseudoHela = pseudoHela.drop(drp, axis=1)

print(controlHela.iloc[0,1])

kmerData = []
for i in range(len(controlHela)):
    kmer = controlHela.iloc[i, 0] 
    kmerData.append([kmer])

    values = controlHela.iloc[i, 1]  

    sig = ""
    for j in range(len(values)):
        if values[j] == '_':
            #convert to int
            kmerData[i].append(int(sig))
            sig = ""

        elif j == (len(values) - 1):
            sig += values[j]
            kmerData[i].append(int(sig))
            sig = ""

        else:
            sig += values[j]
        


pseudoKmerData = []
for i in range(len(pseudoHela)):
    kmer = pseudoHela.iloc[i, 0]
    pseudoKmerData.append([kmer])

    values = pseudoHela.iloc[i, 1]
    sig = ""
    for j in range(len(values)):
        if values[j] == '_':
            #convert to int
            pseudoKmerData[i].append(int(sig))
            sig = ""

        elif j == (len(values) - 1):
            sig += values[j]
            pseudoKmerData[i].append(int(sig))
            sig = ""

        else:
            sig += values[j]

X = []
Xval = []
Y = []
Yval = []


prevIndexes = np.random.choice(len(controlHela), 360, replace=False)
kmerData = np.array(kmerData)[prevIndexes]
print("size of ", len(kmerData))
total = 360 + len(pseudoHela)
indexes = np.random.choice(total, total, replace=False)    


for i in range(len(kmerData)):
    X.append(kmerData[i][0])


for i in range(len(pseudoKmerData)):
    X.append(pseudoKmerData[i][0]) # Now X stores kmerData and pseudoKmerData


#onehot encoding of kmer data
le = preprocessing.LabelEncoder()          
le.fit(X)
print(le.classes_)
X = le.transform(X)
X = X.reshape(-1, 1)

#onehot encode
_, n_features = np.shape(X)
print("&&&&&",n_features)
enc = OneHotEncoder(handle_unknown='ignore',categories=[range(350)]*n_features)# note we replace n_values=350 which is depricated in vrsion 0.2 with categories=[range(350)]*n_features
enc.fit(X)
onehots = enc.transform(X).toarray()
X = onehots

##feature extrextion from unmodified extracted Nanopore signal 
allKmerData=[]
for i in range(len(kmerData)):
    #orginal signal without padding
    allKmerData.append(kmerData[i][1:]) #store rows
    Xval.append([mean(kmerData[i][1:]), median(kmerData[i][1:]), max(kmerData[i][1:]), min(kmerData[i][1:]),get_signal_length(kmerData[i][1:]), np.std(kmerData[i][1:])])
    Yval.append([0])

##feature extrextion from modified extracted Nanopore signal 
for i in range(len(pseudoKmerData)):
    allKmerData.append(pseudoKmerData[i][1:])
    Xval.append([mean(pseudoKmerData[i][1:]), median(pseudoKmerData[i][1:]), max(pseudoKmerData[i][1:]), min(pseudoKmerData[i][1:]),get_signal_length(pseudoKmerData[i][1:]),np.std(pseudoKmerData[i][1:])])
    Yval.append([1])



#combine onehot encoding of kemer features with features extracted from nanopore signal
for i in range(len(Xval)):
    for j in range(len(X[i])):
        Xval[i].append(X[i][j])


#randomize indexes
X = np.array(Xval)[indexes]                   
#X = np.array(X)[indexes]    #when X has onehot encoding of kemer features only 
Y = np.array(Yval)[indexes]
print("pppppppppp",X.shape)
print("jjjjjj",Y.shape)
print("xxxxxxxxxxxxx",type(X))
print("MMMMMMMMMMMMMMM",type(Y))
print(len(X), len(Y))

print("************************",X[0])
print("-------------",Y[0])

########################################################
#To get nanopore signal intenesity 
#get padded signal data
x= signal.signal_data(allKmerData)
print("xxxxxxxxxxxxx",type(x))
print(len(x))
print("##########",x[0])



#concatenate signal intensity features with other features extracted from extracted Nanopore signal
X1=np.concatenate((X,x),axis=1)
print("OOOOOOOOOOOOOOOOOO",X1)

##################apply different ML algorithms###############################

#split the data randomely into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)#Random_State which is an instance used by np.random is the random number generator

#scale training data
X_train= scaler.fit_transform(X_train) 
#scale testing data
X_test= scaler.fit_transform(X_test)


########################################### normal classification with SVM#######################

clf = classifier.fit(X_train,y_train.ravel())



y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
y_prob = y_prob[:,1]
# For model evaluation
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix



print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)
#To get the confusion matrix
print('CF=',confusion_matrix(y_test, y_pred))
l=confusion_matrix(y_test, y_pred)
print('TN',l.item((0, 0)))
print('FP',l.item((0, 1)))
print('FN',l.item((1, 0)))
print('TP',l.item((1, 1)))


#plot learning curve: 
import matplotlib.pyplot as plt
plc. plot_learning_curves(classifier, X_train, y_train, X_test, y_test)

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#plot ROC curve
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt.plot(fpr,tpr)
plt.title("ROC Curve")
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() 


