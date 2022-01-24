import time
import random
import sys, os
sys.path.append(os.pardir)
from common.MNIST_Loader import load_MNIST
from Two_Layer_NeuralNet import Two_Layer_NeuralNet

net = Two_Layer_NeuralNet(784, 50, 10)
(x_train, t_train), (x_test, t_test) = load_MNIST(normalize=True, flatten=True, one_hot_encoding=True)

learning_rate = 0.01
train_data_cnt = x_train.shape[0]
batch_size = 100
epoch = 17
loss_list = []

x = None
t = None

for i in range(epoch):
    for k in range(train_data_cnt // batch_size):
        batch_mask = random.randint(0, train_data_cnt - batch_size)

        x = x_train[batch_mask:batch_mask + batch_size]
        t = t_train[batch_mask:batch_mask + batch_size]

        grad = net.calc_gradient(x, t)
        net.params['W1'] -= (learning_rate * grad['W1'])
        net.params['b1'] -= (learning_rate * grad['b1'])
        net.params['W2'] -= (learning_rate * grad['W2'])
        net.params['b2'] -= (learning_rate * grad['b2'])

    loss = net.loss(x, t)
    loss_list.append(loss)
    print("Epoch/Loss: {}/{}".format(i, loss))
    