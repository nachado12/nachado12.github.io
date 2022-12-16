"""
This module is for training, testing an evaluating classifiers.
"""

from numpy import *
from pylab import *

import util

def trainTest(classifier, X, Y, Xtest, Ytest):
    """
    Train a classifier on data (X,Y) and evaluate on
    data (Xtest,Ytest).  Return a triple of:
      * Training data accuracy
      * Test data accuracy
      * Individual predictions on Xtest.
    """

    classifier.reset()                           # initialize the classifier
    classifier.train(X, Y);                      # train it
    
    Ypred = classifier.predictAll(X);            # predict the training data
    trAcc = mean((Y     >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    Ypred = classifier.predictAll(Xtest);           # predict the training data
    teAcc = mean((Ytest >= 0) == (Ypred >= 0));     # check to see how often the predictions are right

    print ("Training accuracy %g, test accuracy %g" % (trAcc, teAcc))

    return (trAcc, teAcc, Ypred)

def trainTestSet(classifier, dataset):
    trainTest(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def learningCurve(classifier, X, Y, Xtest, Ytest):
    """
    Generate a learning curve by repeatedly halving the amount of
    training data until none is left.

    We return a triple containing:
      * The sizes of data sets we trained on
      * The training accuracies at each level
      * The test accuracies at each level
    """

    N = X.shape[0]             # how many total points?
    M = int(ceil(log2(N)))     # how many classifiers will we have to train?

    dataSizes = zeros(M)
    trainAcc  = zeros(M)
    testAcc   = zeros(M)
    
    for i in range(1, M+1):    # loop over "skip lengths"
        # select every 2^(M-i)th point
        ids = arange(0, N, 2**(M-i))
        Xtr = X[ids, :]
        Ytr = Y[ids]

        # report what we're doing
        print ("Training classifier on %d points..." % ids.size)

        # train the classifier
        (trAcc, teAcc, Ypred) = trainTest(classifier, Xtr, Ytr, Xtest, Ytest)
        
        # store the results
        dataSizes[i-1] = ids.size
        trainAcc[i-1]  = trAcc
        testAcc[i-1]   = teAcc

    return (dataSizes, trainAcc, testAcc)

def learningCurveSet(classifier, dataset):
    return learningCurve(classifier, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def hyperparamCurve(classifier, hpName, hpValues, X, Y, Xtest, Ytest):
    M = len(hpValues)
    trainAcc = zeros(M)
    testAcc  = zeros(M)
    for m in range(M):
        # report what we're doing
        print ("Training classifier with %s=%g..." % (hpName, hpValues[m]))
        
        # train the classifier
        classifier.setOption(hpName, hpValues[m])
        classifier.reset()
        (trAcc, teAcc, Ypred) = trainTest(classifier, X, Y, Xtest, Ytest)

        # store the results
        trainAcc[m] = trAcc
        testAcc[m]  = teAcc

    return (hpValues, trainAcc, testAcc)

def hyperparamCurveSet(classifier, hpName, hpValues, dataset):
    return hyperparamCurve(classifier, hpName, hpValues, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def plotCurve(titleString, res):
    figure(figsize=(16, 10), dpi=100)
    plot(res[0], res[1], 'b-',
         res[0], res[2], 'r-')
    legend( ('Train', 'Test') )
    ylabel('Accuracy')
    title(titleString)
    show()
