"""
Support Vector Classifier
David Felipe Alvear Goyes
Artificial Intelligence Course 2020 - I
Columbia University
04/2020
"""

import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


class SVM(object):

    def __init__(self, data, methods_list):
        # Divide the data in train and test
        self.data_train, self.data_test = train_test_split(data, test_size = 0.40)  # [x, y, label]
        self.methods = methods_list
        
    def run(self):
        # Run a classification for the data given and methods selected
        x_data = self.data_train[:,:-1]
        y_data = self.data_train[:,-1:]
        x_test = self.data_test[:,:-1]
        y_test = self.data_test[:,-1:]
        results = []
        for method in self.methods:
            print(method)
            # First Build the model to use
            classifier = self.build_svm(method)
            # fit the classifier
            classifier.fit(x_data, y_data.ravel())
            # Get the best classifier
            best_estimator = classifier.best_estimator_
            # Get the best training score
            best_score = classifier.best_score_
            # Compute best test score
            test_score = best_estimator.score(x_test, y_test)
            # Show information
            print(method + " Train score : " + str(best_score) + "  Test score : " + str(test_score))
            results.append([method, best_score, test_score])
        
        return results

    def build_svm(self, method = "linear"):
        # Search best values of regularization
        #svm_init = SVC()
        if method == "svm_linear":
            svm_init = SVC()
            parameters = {'kernel':('linear',), 'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
        
        elif method == "svm_polynomial":
            svm_init = SVC()
            parameters = {'kernel':('poly',), 'C' : [0.1, 1, 3], 'degree' : [4, 5, 6], 'gamma' : [0.1, 0.5]}

        elif method == "svm_rbf":
            svm_init = SVC()
            parameters = {'kernel':('rbf',), 'C' : [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma' : [0.1, 0.5, 1, 3, 6, 10]}

        elif method == "logistic":
            svm_init = LogisticRegression(solver='lbfgs')
            parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}

        elif method == "knn":
            svm_init = KNeighborsClassifier()
            parameters = {'n_neighbors':np.arange(1,51), 'leaf_size':np.arange(5,61)}

        elif method == "decision_tree":
            svm_init = DecisionTreeClassifier()
            parameters = {'max_depth':np.arange(1,51), 'min_samples_split':np.arange(2,11)}
        
        elif method == "random_forest":
            svm_init = RandomForestClassifier(n_estimators=100)
            parameters = {'max_depth':np.arange(1,51), 'min_samples_split':np.arange(2,11)}

        classifier = GridSearchCV(svm_init, parameters, cv=5, iid=False)

        return classifier


class data_utils(object):
    def __init__(self, input, output):
        self.input_string = input
        self.output = output
    
    def load_data(self):  
        """
            load_data - Read csv file with the linear dataset 
            - input_string : name of the csv file to read
            - return array with the data loaded
        """
        input_data = pd.read_csv(self.input_string, sep=',')
        return input_data.values
    
    def scatter_plot(self, data):
        data1 = data[data[:,2]==1]
        data2 = data[data[:,2]==0]

        plt.plot(data1[:,0], data1[:,1], 'ro')
        plt.plot(data2[:,0], data2[:,1], 'bo')
        plt.show()

    def output_file(self, list_betas):
        pd.DataFrame(list_betas).to_csv(self.output, header=None, index=None)


def main():
    ## First - load the input1.csv
    sm = sys.argv[1].lower()
    input_name = sys.argv[1].split(" ")[0]
    output_name = sys.argv[2].split(" ")[0]

    methods = ["svm_linear", "svm_polynomial", "svm_rbf", "logistic", "knn", "decision_tree", "random_forest"]

    data_manager = data_utils(input_name, output_name)
    raw_data = data_manager.load_data()
    #data_manager.scatter_plot(raw_data) # Grapgh the data

    supportVectMach = SVM(raw_data, methods)
    results = supportVectMach.run()

    ## save results
    data_manager.output_file(results)

if __name__ == '__main__':
    main()