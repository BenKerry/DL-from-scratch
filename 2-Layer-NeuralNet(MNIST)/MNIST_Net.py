import numpy as np

class MNIST_Net:
    def __init__(self, input_size:int, hidden_size:int, output_size:int, weight_init_std=0.01):
        self.params = {}

        self.params['W1'] = weight_init_std / np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std / np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        L1 = sigmoid(np.dot(x, W1) + b1)
        y = softmax(np.dot(L1, W2) + b2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return CEE(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        y = np.argmax(x)
        t = np.argmax(t)

        return np.sum(y == t) / float(x.shape[0])

    def calc_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = calc_gradient(loss_W, self.params['W1'])
        grads['b1'] = calc_gradient(loss_W, self.params['b1'])
        grads['W2'] = calc_gradient(loss_W, self.params['W2'])
        grads['b2'] = calc_gradient(loss_W, self.params['b2'])

        return grads