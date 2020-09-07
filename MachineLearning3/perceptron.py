import numpy as np

class perceptron(object):
    def __init__(self, input_size = 2):
        """
        Create a perceptron with the given parameters
            - input_size : variable to input to the neuron
        """
        # intitialize the weights
        self.weights = np.random.rand(input_size + 1)
        
    def neuron(self, inputs = []):
        """
        neuron - Function to activate the neuron for a given data.
            - inputs : Data to fit in the neuron
        """
        # Data
        inputs = np.hstack((np.ones((len(inputs), 1), dtype=np.int), inputs))
        # Compute 
        weighted_sum = self.weighted_sum(inputs)
        output = self.sigmoid_activation(weighted_sum)

        return output
        
    def weighted_sum(self, data):
        # Create the weighted sum iterator
        for line in data:
            W_sum += self.weights * line
        return W_sum
    
    def sigmoid_activation(self, weighted_sum):
        G_ativation = 1 / (1 + np.exp(-weighted_sum))
        return G_ativation
        
