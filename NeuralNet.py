import numpy as np
import pandas as pd
from numpy.core._multiarray_umath import ndarray

    # The NeuralNet class upon initialization, creates a series of matrices representing the weights.
	# It calls the gen_weights function to accomplish this.
	# Although the testing portion is set up to analyze handwritten digits (0-9). The methods for training are generalized enough to 
	# allow for a wider range of data.
class NeuralNet:
    def __init__(self, train_data, train_labels, test_data, test_labels, epoch, h_nodes, lr):
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.inputs = None
        self.target = None

        # The output of the hidden layer post-sigmoid function
        self.h_act = None
        self.o_act = None
        self.lr = lr
        self.epoch = epoch
        # Include a value/array to compare output to target?
        self.whi = None
        self.woh = None

        self.i_nodes = len(train_data.columns)
        self.h_nodes = h_nodes
        self.o_nodes = train_labels.nunique()[0]

        self.gen_weights()

    # Functions
    # We need to generate weights for all whi and woh combinations
    # By using ndarray we can use np.dot to multiply matrices
    # When sizing the array, we must make sure to take the bias into account.
    # There must be one additional column so that the number of columns is
    # The number of inputs + 1.
    # This is also true for the weights between the hidden and output layers.
    #   /                    \  /       \
    #   | w11 w12 w13 ... w1b | | in0   |
    #   | w21 w22 w23 ... ... | | in1   |
    #   | w31 w32 w33 ... ... |*| in2   |
    #   | ... ... ... ... ... | | ...   |
    #   | wn1 ... ... ... wnb | | bias  |
    #   \                    /  \       /
    def gen_weights(self):
        # Create matrices for input -> hidden weights, and hidden -> output weights, whi and woh respectively
        # Add 1 weight to account for the biases.

        # total number of weights for each layer
        ih_size = (self.i_nodes + 1) * self.h_nodes
        ho_size = (self.h_nodes + 1) * self.o_nodes

        # Randomize an array of floats between two values.
        ih_raw = np.random.uniform(-0.05, 0.05, ih_size)
        ho_raw = np.random.uniform(-0.05, 0.05, ho_size)

        # Shape the arrays into 2D arrays of shape h nodes, i nodes + 1
        self.whi = ih_raw.reshape(self.h_nodes, self.i_nodes + 1)
        self.woh = ho_raw.reshape(self.o_nodes, self.h_nodes + 1)

    # input_values should be the same size as i_nodes
    def prop_forward(self, i_ins):
        # Create a 2d ndarray with 1 additional value for bias
        i_ins.append(1)
        i_ins = np.asarray(i_ins)
        i_ins = i_ins.reshape((len(i_ins)), 1)

        h_outs = np.dot(self.whi, i_ins)

        # Add hidden layer bias
        h_outs = np.append(h_outs, [1])
        self.h_act = sigmoid(h_outs)
        self.h_act = self.h_act.reshape(self.h_nodes + 1, 1)
        print('Weights:\n', self.woh, '\n*\n', 'Hidden Node Outputs:\n', self.h_act.T)
        outs = np.dot(self.woh, self.h_act)
        self.o_act = sigmoid(outs)

        # print(self.h_act, '\n', self.o_act,'\n****')
        return self.o_act

    # To back propogate we use the following functions:
    # Any time the equation requires a Sigma, we can use dot product because it is a scalar operation.
    # .T is a method which transposes a matrix.
    def back_propagate(self):

        self.inputs = np.asarray(self.inputs)
        self.inputs = self.inputs.reshape((len(self.inputs)), 1)
        # calculate error
        # In output
        loss = error(self.target, self.o_act)
        # print(loss)
        e_o = loss.T

        # In hidden
        e_h = np.dot(e_o, self.woh).T * d_sigmoid(self.h_act)

        d_woh = np.dot(self.h_act, e_o).T * self.lr
        # There is no xji for input -> the bias of the hidden layer
        d_whi = np.dot(e_h[:-1], self.inputs.T) * self.lr
        self.woh += d_woh
        self.whi += d_whi

    # The training method, commits forward propogation.
    def train(self, limit=None):
        instances = self.train_data.shape[0]

        if limit:
            instances = limit

        print('Training based on ', self.epoch, ' epochs')
        for i in range(self.epoch):
            for j in range(instances):
                # print('[', i, ',', j, ']')
                self.inputs = self.train_data.iloc[j].tolist()
                self.target = self.train_labels.iloc[j]

                self.prop_forward(self.inputs)
                self.back_propagate()

                # print(self.woh)
            print('Epoch ', i+1, ' complete.')
        print('Training Complete')

    # Tests data, will limit iterations if queries is passed
    def test(self, queries=None):
        correct = 0
        instances = self.test_data.shape[0]

        if queries:
            instances = queries

        for i in range(instances):
            # print(self.test_labels)
            out_arr = self.prop_forward(self.test_data.iloc[i].tolist())
            prediction: ndarray = np.where(out_arr == np.amax(out_arr))
            act_arr = create_target(self.test_labels.iloc[i])
            actual = np.where(act_arr == np.amax(act_arr))
            # print(prediction[0][0], ',', actual[0][0])
            if prediction[0][0] == actual[0][0]:
                # print(prediction[0][0], ',', actual[0][0])
                correct += 1

        print("---Results---")
        print('Correct Classifications: ', correct)
        print('Incorrect Classifications: ', instances - correct)
        print('Accuracy: ', correct / instances * 100, '%')


# This is our activation function.
def sigmoid(n):
    return 1 / (1 + np.exp(n * -1))

#derivative of the sigmoid function, to facilitate backpropogation.
def d_sigmoid(n):
    return sigmoid(n) * (1 - sigmoid(n))


# Creates a list of 10 values initialized to 0.
# y is the class label.
def create_target(y):
    target = [0.01] * 10
    target[int(y)] = 0.99
    target = np.reshape(np.asarray(target), (len(target), 1))
    return target


# Returns an array of errors corresponding to the output nodes
def error(target_value, output):
    return d_sigmoid(output) * (create_target(target_value) - output)


def main():
    training_data = pd.read_csv('training60000.csv', header=None)
    training_labels = pd.read_csv('training60000_labels.csv', header=None)
    testing_data = pd.read_csv('testing10000.csv', header=None)
    testing_labels = pd.read_csv('testing10000_labels.csv', header=None)

    neural_net = NeuralNet(training_data, training_labels, testing_data, testing_labels, 15, 16, 0.005)

    # You may pass a value into train or test to limit the number of instances
    neural_net.train()
    neural_net.test()


if __name__ == '__main__':
    main()
