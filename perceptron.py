import pandas as pd
import numpy as np

class Perceptron(object):
    def __init__(self,learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def fit(self,X,y):
        self.weights = np.random.random((1 + X.shape[1]))
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for i,expected in zip(X,y):
                update = self.learning_rate * (expected - self.predict(i))
                self.weights[1:] += update * i
                self.weights[0] += update
                errors += int(update != 0.0)
                self.errors.append(errors)
    
    def net_input(self,X):
        output = np.dot(X, self.weights[1:]) + self.weights[0]
        return output
    def predict(self, X):
        return self.sigmoid(self.net_input(X))

    def sigmoid(self,X):
        return np.exp(-np.logaddexp(0, -X))