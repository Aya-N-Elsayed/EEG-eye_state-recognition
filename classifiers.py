from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import *
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import time
class classifiers():
  
    def __init__(self, classifier, dataset,scaling ,x=0, y=0,X_train =0, X_test = 0, y_train =0, y_test=0,classes=[]):
        self.classifier = classifier
        self.dataset = dataset
        self.x = x
        self.y = y
        self.scaling = scaling
        self.classes = classes
        self.X_train = X_train
        self.X_test = X_test 
        self.y_train = y_train  
        self.y_test = y_test

    def read_dataset(self):

        try:
            data = pd.read_csv(self.dataset, header=None)
            self.x = data .iloc[:, 0:(data.shape[1]-1)].values
            self.y = data .iloc[:, -1].values
        except:
            print("Data set not found")

                
        print('Class labels:', np.unique( self.y ))
        self.splitting_data()
        if (self.scaling):
            self.standardizing()
        
    
    def splitting_data (self):
        # Splitting data into 70% training and 30% test data:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=1, stratify=self.y)
        print('Labels counts in y:', np.bincount(self.y))
        print('Labels counts in y_train:', np.bincount(self.y_train))
        print('Labels counts in y_test:', np.bincount(self.y_test))

    def standardizing (self):
        # Standardizing the features:
        sc = StandardScaler()
        sc.fit(self.X_train)
        X_train_std = sc.transform(self.X_train)
        X_test_std = sc.transform(self.X_test)
        self.X_train = X_train_std 
        self.X_test = X_test_std 
    def run_classifier(self):
        if self.classifier == 'Perceptron':
            ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
            start_time = time.time()
            ppn.fit(self.X_train, self.y_train)
            End_time = time.time()
            y_pred = ppn.predict(self.X_test)
            print('Accuracy of training: %.2f' % ppn.score(self.X_train, self.y_train))
        elif self.classifier == 'LinearSVM':
            svm = SVC(kernel='linear', C=2.0, random_state=1)
            start_time = time.time()
            svm.fit(self.X_train, self.y_train)
            End_time = time.time()
            y_pred = svm.predict(self.X_test)
            print('Accuracy of training: %.2f' % svm.score(self.X_train, self.y_train))
        elif self.classifier == 'RBFSVM':
            svm = SVC(kernel='rbf', random_state=1, gamma=1, C=1.0)
            start_time = time.time()
            svm.fit(self.X_train, self.y_train)
            End_time = time.time()
            y_pred = svm.predict(self.X_test)
            print('Accuracy of training: %.2f' % svm.score(self.X_train, self.y_train))
        elif self.classifier == 'DT':
            tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
            start_time = time.time()
            tree.fit(self.X_train, self.y_train)
            End_time = time.time()
            y_pred = tree.predict(self.X_test)
            print('Accuracy of training: %.2f' % tree.score(self.X_train, self.y_train))
        elif self.classifier == 'KNN':
            knn = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
            start_time = time.time()
            knn.fit(self.X_train, self.y_train)
            End_time = time.time()
            y_pred = knn.predict(self.X_test)
            print('Accuracy of training: %.2f' % knn.score(self.X_train, self.y_train))
        else:
            print("Error not a valid classifier")
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())
        print('Accuracy OF Testing: %.2f'  %accuracy_score(self.y_test, y_pred))
        #plot_confusion_matrix(model, X_test, y_test, cmap='GnBu')
        #plt.show()
        print('Precision: %.3f' %precision_score(self.y_test, y_pred))
        print('Recall: %.3f' %recall_score(self.y_test, y_pred))
        print('F1: %.3f' %f1_score(self.y_test, y_pred))
        print (" Excuation time", (End_time- start_time))   
