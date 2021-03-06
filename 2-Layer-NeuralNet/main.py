import time
import random
import pickle
import sys, os
sys.path.append(os.pardir)
from common.MNIST_Loader import load_MNIST
from Two_Layer_NeuralNet import Two_Layer_NeuralNet

net = Two_Layer_NeuralNet(784, 50, 10)
(x_train, t_train), (x_test, t_test) = load_MNIST(normalize=True, flatten=True, one_hot_encoding=True)

learning_rate = 0.01
batch_size = 100
epoch = 17

loss_list = []
acc_list = []
elapsed_time_list = []

train_data_cnt = x_train.shape[0]
iter_for_epoch = train_data_cnt // batch_size

x = None
t = None

s_training = time.time()

for i in range(epoch):
    for k in range(iter_for_epoch):
        s_batch = time.time()
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

        accuracy = net.accuracy(x, t)
        acc_list.append(accuracy)

        elapsed_time = time.time() - s_batch
        elapsed_time_list.append(elapsed_time)

        ETA_second = (sum(elapsed_time_list) / len(elapsed_time_list)) * (epoch - i) * (iter_for_epoch - k)

        print("__________Epoch {}/{}__________".format(i + 1, epoch))
        print("***** Batch: {}/{} *****".format(k + 1, iter_for_epoch))
        print("Loss/Accuracy: {}/{}%".format(loss, accuracy * 100))
        print("Elapsed Time(1 Batch): {}s".format(int(elapsed_time)))
        if ETA_second < 60:
            print("ETA: {}m".format((ETA_second / 60) % 60))
        else:
            print("ETA: {}h {}m".format(ETA_second // 3600, (ETA_second / 60) % 60))

net.model_infos['training_time'] = "{}m".format((time.time() - s_training) / 60)
net.model_infos['loss_list'] = loss_list
net.model_infos['accuracy_list'] = acc_list

if not os.path.isdir("model"):
    os.mkdir("model")

with open("model/({})MNIST_model.pkl".format(time.strftime("%y-%m-%d")), "wb") as fp:
    pickle.dump(net, fp)