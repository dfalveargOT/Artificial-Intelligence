"""
Linear Regression Algorithm
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

class linear_regression(object):
    def __init__(self, data, labels, lr, iterations):
        self.data = data
        self.labels = labels
        self.lr = lr
        self.iterations = iterations

    def run(self):
        betas = np.zeros(self.data.shape[1])
        log_betas = []

        # iterate over all the learning rates
        for lrate in self.lr:
            counter = 0
            betas = np.zeros(self.data.shape[1]) # reset betas 

            # Gradient descent for the given learning rate
            for counter in range(self.iterations+1):
                betas = self.gradient_descent(self.data, self.labels, betas, lrate)

            log_betas.append([lrate, counter, betas[0], betas[1], betas[2]])
        
        return log_betas
    
    def gradient_descent(self, data, labels, betas, lr):
        # Calculate the derivative of the risk function
        functionx = np.dot(data, betas)
        temp1 = (functionx[:,None]-labels).flatten()
        loss = (data*temp1[:,None]).sum(axis=0)
        dRisk = lr * loss / (data.shape[0])

        # Update betas
        betas = betas - dRisk
        return betas

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
        input_data = pd.read_csv(self.input_string, sep=',',header=None)

        return input_data.values
    
    def normalization(self, data):
        """
        Data - age, weight, height
        return - normalized data 
        """
        # Nomrmalize the data
        #mean = np.mean(data, axis=0) #[age_m, we_m, hei_m]
        mean = data.mean(axis=0)
        # Standard deviation
        #std = np.std(data, axis=0) #[age_std, we_std, hei_std]
        std = data.std(axis=0)
        data = (data - mean[None,:]) / std[None,:]
        return data
        

    def output_file(self, list_betas):
        pd.DataFrame(list_betas).to_csv(self.output, header=None, index=None)


def main():
    ## First - load the input1.csv
    sm = sys.argv[1].lower()
    input_name = sys.argv[1].split(" ")[0]
    output_name = sys.argv[2].split(" ")[0]

    lrList = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.67]
    #iterationMaxList = [100, 100, 100, 100, 100, 100, 100, 100, 100, 58]


    data_manager = data_utils(input_name, output_name)
    raw_data = data_manager.load_data()
    data = data_manager.normalization(raw_data[:,:-1]) # Normalize data

    #data = data[:,:-1] # extract X inputs from the data
    labels = raw_data[:,-1:] # extract labels from the data
    datax = np.c_[np.ones(raw_data.shape[0]), data] # add one column at end for w0

    # linearRegression
    linearRegression = linear_regression(datax, labels, lrList, 100)
    log_betas = linearRegression.run()

    ## save results
    data_manager.output_file(log_betas)
    #data_maneger.graph(raw_data, final_weights)

if __name__ == '__main__':
    main()