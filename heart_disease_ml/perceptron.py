"""
In perceptron.py, you will implement the perceptron algorithm for
binary classification.  You will implement both the vanilla perceptron
updates as well as the averaged perceptron updates.
"""

from tkinter.messagebox import YESNO
from numpy import *

from binary import *

class Perceptron(BinaryClassifier):
    """
    This class defines the perceptron implementation of a binary
    classifier.  See binary.py for details on the abstract class that
    this implements.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  You probably need to (at
        least) keep track of a weight vector and a bias.  We'll just
        call the 'reset' function to do this for us.

        We will also want to compute simple statistics about how the
        training of the perceptron is going.  In particular, you
        should keep track of how many updates have been made total.
        """

        BinaryClassifier.__init__(self, opts)
        self.opts = opts
        self.reset()

    def reset(self):
        """
        Reset the internal state of the classifier.
        """

        self.weights = 0    # our weight vector
        self.bias    = 0    # our bias
        self.numUpd  = 0    # number of updates made

        self.u = 0
        self.beta = 0
        self.counter = 0

    def online(self):
        """
        Our perceptron is online
        """
        return True

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)   +  ", b=" + repr(self.bias)

    def predict_voted(self, X):
        if self.numUpd == 0:
            return 0    # failure

        prediction = 0
        for i in range(self.num_perceptrons):
            prediction += (dot(self.vote_weights[i], X) + self.vote_bias[i]) * self.votes[i]

        return prediction

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the margin at this point.
        Semantically, a return value <0 means class -1 and a return
        value >=0 means class +1
        """

        if self.numUpd == 0:
            return 0          # failure
        else:
            return dot(self.weights, X) + self.bias   # this is done for you!
            # return self.predict_voted(X)

    def nextExample(self, X, Y):
        """
        X is a vector training example and Y is its associated class.
        We're guaranteed that Y is either +1 or -1.  We should update
        our weight vector and bias according to the perceptron rule.
        """

        # check to see if we've made an error
        
        # print(Y * self.predict(X), "OKAY END")
        if Y * self.predict(X) <= 0:   ### SOLUTION-AFTER-IF
            self.numUpd  = self.numUpd  + 1

            # perform an update
            self.weights = self.weights + Y * X
            self.bias    = self.bias + Y

            # tracking bias for averaged perceptron
            self.u = self.u + self.counter * Y * X
            self.beta = self.beta + self.counter * Y
        
        self.counter += 1



    def nextIteration(self):
        """
        Indicates to us that we've made a complete pass through the
        training data.  This function doesn't need to do anything for
        the perceptron, but might be necessary for other classifiers.
        """

        return   # don't need to do anything here
        

    def getRepresentation(self):
        """
        Return a tuple of the form (number-of-updates, weights, bias)
        """
        if (self.opts['averaged']):
            averaged_w = (1/self.counter) * self.u
            averaged_b = (1/self.counter) * self.beta
            for i in range(len(averaged_w)):
                averaged_w[i] = int(averaged_w[i])
            self.weights = self.weights - averaged_w
            self.bias = self.bias - averaged_b
            return (self.numUpd, self.weights, self.bias)
        else:
            return (self.numUpd, self.weights, self.bias)