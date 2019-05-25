from lib.matrix import Matrix
from math import exp
import numpy


def sigmoid(x):
    return 1 / (1 + exp(-x))


def dsigmoid(y):
    return y * (1 - y)


class NeuralNetwork:
    '''
    Simple Neural Network with three layers
    '''

    def __init__(self, numI, numH, numO):
        self.numI = numI
        self.numH = numH
        self.numO = numO

        self.weights_ih = Matrix(numH, numI)
        self.weights_ho = Matrix(numO, numH)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = Matrix(numH, 1)
        self.bias_o = Matrix(numO, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

        self.learning_rate = 0.1

    @staticmethod
    def from_trained(info):
        '''
        Creates a Neural Network from the information
        of another one
        '''
        numI = info['dims'][0]
        numH = info['dims'][1]
        numO = info['dims'][2]
        nn = NeuralNetwork(numI, numH, numO)
        nn.weights_ho.data = info['weights_ho']
        nn.weights_ih.data = info['weights_ih']
        nn.bias_h.data = info['bias_h']
        nn.bias_o.data = info['bias_o']

        return nn

    def predict(self, input_array):
        '''
        Predicts the answers based on some input
        '''
        # Generating the hidden outputs
        inputs = Matrix.from_array(input_array)
        hidden = Matrix.static_mult(self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # Activation function!
        hidden.map(sigmoid)

        # Generating the output's output
        output = Matrix.static_mult(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(sigmoid)

        # Sending back to caller!
        return output.to_array()

    def train(self, input_array, target_array):
        '''
        Trains the NN based on some input_data and
        target_data
        '''
        # Generating the hidden outputs
        inputs = Matrix.from_array(input_array)
        hidden = Matrix.static_mult(self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # Activation function!
        hidden.map(sigmoid)

        # Generating the output's output
        outputs = Matrix.static_mult(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)

        # Convert array to matrix object
        targets = Matrix.from_array(target_array)

        # Calculate error
        # error = targets - outputs

        output_errors = Matrix.subtract(targets, outputs)
        # gradient = outputs * (1 - outputs)

        # Calculate gradient
        gradient = Matrix.static_map(outputs, dsigmoid)
        gradient.mult(output_errors)
        gradient.mult(self.learning_rate)

        # Calculate deltas
        hidden_T = Matrix.transpose(hidden)
        weight_ho_deltas = Matrix.static_mult(gradient, hidden_T)

        # Adjust the weigts by deltas
        self.weights_ho.add(weight_ho_deltas)
        self.bias_o.add(gradient)

        # Calculate the hidden layer errors
        weights_ho_T = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.static_mult(weights_ho_T, output_errors)

        # Calculate hidden gradient
        hidden_gradient = Matrix.static_map(hidden, dsigmoid)
        hidden_gradient.mult(hidden_errors)
        hidden_gradient.mult(self.learning_rate)

        # Calculate input to hidden deltas
        inputs_T = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.static_mult(hidden_gradient, inputs_T)
        self.weights_ih.add(weight_ih_deltas)
        self.bias_h.add(hidden_gradient)

    def save(self):
        '''
        Saves the information of the NN to a .npz file, so it
        can be loaded later to create another NN
        '''
        dims = numpy.array([self.numI, self.numH, self.numO])
        weights_ih = numpy.array(self.weights_ih.to_array()).reshape(
            (self.numH, self.numI))
        weights_ho = numpy.array(self.weights_ho.to_array()).reshape(
            (self.numO, self.numH))
        bias_h = numpy.array(self.bias_h.to_array()).reshape((self.numH, 1))
        bias_o = numpy.array(self.bias_o.to_array()).reshape((self.numO, 1))

        numpy.savez("digit_predictor_NN.npz", dims=dims,
                    weights_ih=weights_ih, weights_ho=weights_ho, bias_h=bias_h, bias_o=bias_o)
