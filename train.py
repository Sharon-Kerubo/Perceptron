from perceptron import Perceptron
import numpy as np

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == np.around(y_pred)) / len(y_true)
    return accuracy
with open("heart.data.txt") as f:
    data = np.array(list(list(map(lambda x: float(x), line.strip().split(' '))) for line in f.readlines()))
    X = data[:, 0:-1]
    y = data[:, -1]
    y -= 1

D_train = int(len(data) * 0.7)
X_train = X[:D_train]
y_train = y[:D_train]
X_test = X[D_train:]
y_test = y[D_train:]

prcptrn = Perceptron(0.01, 1000)
prcptrn.fit(X_train, y_train)
predictions = prcptrn.predict(X_test)
print(accuracy(y_test, predictions))