"""
David Felipe Alvear Goyes
Artificial Intelligence Course 2020 - I
Columbia University
04/2020
"""
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

class perceptron(object):
    def __init__(self, input_size, lr = 0.3):
        """
        Create a perceptron with the given parameters
            - input_size : variable to input to the neuron
        """
        # intitialize the weights
        self.weights = np.zeros(input_size)
        self.learning_rate = lr
        self.error = 0
        
    def neuron(self, inputs, label):
        """
        neuron - Function to activate the neuron for a given data.
            - inputs : Data to fit in the neuron
        """
        # Data

        # Compute 
        weighted_sum = self.weighted_sum(inputs)
        #output = self.step_function(weighted_sum)
        
        # save result for update weights
        if weighted_sum == 0:
            weighted_sum = -1

        #self.error = (label - output)*output*(1-output)
        return weighted_sum
        
    def weighted_sum(self, data):
        # Create the weighted sum iterator  
        W_sum = np.dot(self.weights, data)
        return W_sum
    
    def step_function(self, weighted_sum):
        output = 0
        if weighted_sum > 0:
            output = 1
        else:
            output = -1
        return output

    def sigmoid_activation(self, weighted_sum):
        G_ativation = 1 / (1 + np.exp(-weighted_sum))
        return G_ativation

    def update_weights(self, inputs, label):
        
        #self.weights = self.weights + self.error * inputs
        self.weights = self.weights + label * inputs
        


class network(object):
    def __init__(self, size_input=2, size_output=1, error = 0.1):
        ## create the network
        self.neuron1 = perceptron(size_input) # Neuron 1
        self.log_weights = []#np.zeros((1,3))
        self.min_error = error

    def fit(self, iterations, training_data, labels):
        # 1 - Feedforward : propagate example x throught the network counter < iterations or
        error = 1
        counter = 0
        while error:
            error = 0
            for train_data, label in zip(training_data, labels):
                output = self.neuron1.neuron(train_data, label)
                
                # update weights
                if label * output <= 0: 
                    # error was occurred
                    self.neuron1.update_weights(train_data, label)
                    error = 1
                else:
                    final_weights = self.neuron1.weights

                # save weights updated
                self.log_weights.append([self.neuron1.weights[1], self.neuron1.weights[2], 
                                            self.neuron1.weights[0]])
            counter += 1
            
        # Save the weights updated
        return self.log_weights, final_weights
        

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

    def output_file(self, list_weights):
        pd.DataFrame(list_weights).to_csv(self.output, header=None, index=None)
    
    def graph(self, data, weights):

        data1 = data[data[:,2]==1]
        data2 = data[data[:,2]==-1]

        plt.plot(data1[:,0], data1[:,1], 'ro')
        plt.plot(data2[:,0], data2[:,1], 'bo')
        #Plot decision boundary    
        x = np.linspace(min(data[:,0]),max(data[:,0]))
        
        y = (-weights[0]*x - weights[2])/weights[1]    
        plt.plot(x,y)
        plt.show()


def main():
    ## First - load the input1.csv
    sm = sys.argv[1].lower()
    input_name = sys.argv[1].split(" ")[0]
    output_name = sys.argv[2].split(" ")[0]

    data_maneger = data_utils(input_name, output_name)
    raw_data = data_maneger.load_data()

    datax = raw_data[:,:-1] # extract X inputs from the data
    labels = raw_data[:,-1:] # extract labels from the data
    datax = np.c_[datax, np.ones(raw_data.shape[0])] # add one column at end for bias

    ## Pass throught perceptron
    Network = network(size_input=raw_data.shape[1], size_output=1)
    weights, final_weights = Network.fit(1000, datax, labels)

    ## save results
    data_maneger.output_file(weights)
    #data_maneger.graph(raw_data, final_weights)

if __name__ == '__main__':
    main()