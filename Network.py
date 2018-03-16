# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:52:30 2018

@author: Jeff Budge

Neural Network class

INPUTS
training data
initial weights
initial biases
"""

# Third-party libraries
import numpy as np
#import random
import pickle
from pylab import shape
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class Network(object):
    
    def __init__(self, layers):
        self.model = MLPClassifier(solver='adam', hidden_layer_sizes=layers)
        self.layers = layers
        
    def train(self, training_data, labels):
        scaler = StandardScaler()
        scaler.fit(training_data)
        self.model.fit(scaler.transform(training_data), labels)
        self.weights = self.model.coefs_
        self.biases = self.model.intercepts_
        self.means = scaler.mean_
        self.variance = scaler.var_
        self.scale = scaler.scale_
        
            
    def load(self, weights_file):
        f = pickle.load(open(weights_file))
        self.weights = f['weights']
        self.biases = f['biases']
        self.layers = []
        self.means = f['means']
        self.variance = f['variance']
        self.scale = f['scale']
        for i in self.weights:
            self.layers.append(shape(i)[0])
        print "Pickle loaded."
            
    def save(self, file_loc):
        weights_file = {'weights': self.weights, 'biases': self.biases, 'means': self.means, 'variance': self.variance, 'scale':self.scale}
        pickle.dump(weights_file, open(file_loc, "w"))
        print "Pickle dumped."
        
    def run(self, inputs, percentages = False):
        sz = shape(inputs)[0] if inputs.ndim > 1 else 1
        ret = np.zeros((sz,))
        percs = np.zeros((sz, shape(self.weights[-1])[1]))
        for i in range(sz):
            if inputs.ndim == 1:
                curr_inp = ((inputs - self.means) / self.variance) * self.scale
            else:
                curr_inp = ((inputs[i, :] - self.means) / self.variance) * self.scale
            for l_idx in range(len(self.weights)): 
                curr_weights = self.weights[l_idx]
                curr_bias = self.biases[l_idx]
                curr_inp = relu(curr_weights.T.dot(curr_inp)+curr_bias)
            output = np.exp(curr_inp) / np.sum(np.exp(curr_inp)) #softmax function for evaluation of state
            ret[i] = np.where(output == output.max())[0][0]
            if percentages:
                percs[i, :] = output
        if percentages:
            return ret, percs
        else:
            return ret
        
    #Only used to validate the output from the manual function. We all good here.
    def model_run(self, inputs):
        return self.model.predict(inputs)
   
#Activation functions     
def htan(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def relu(z):
    return np.maximum(0, z)
        