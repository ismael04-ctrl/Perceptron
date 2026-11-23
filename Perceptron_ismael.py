import numpy as np


class Perceptron:
    def __init__(self, eta=0.001, n_iter=1000, semilla=667):
        self.eta = eta
        self.n_iter = n_iter
        self.semilla = semilla

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        generador_aleatorio = np.random.RandomState(self.semilla)
        self.w_ = generador_aleatorio.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = 0.0
        self.errores_ = []
        for _ in range(self.n_iter):
            errores_epoca = 0
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                incremento = self.eta * (target - pred)
                self.w_ += incremento * xi
                self.bias_ += incremento
                if incremento != 0.0:
                    errores_epoca += 1
            self.errores_.append(errores_epoca)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.bias_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
