import os
import time
import gzip
import pickle
import numpy as np
import urllib.request as req

def _log(msg):
    print("[Log][{0}]".format(time.strftime("%H:%m:%S")) + msg)

def _download_dataset(fname, save_name):
    _log("[_download_dataset()] Starting Dataset Download...({0})".format(fname))
    fpath = "MNIST_Raw/"

    if not os.path.isdir(fpath):
        os.mkdir(fpath)
    else:
        if os.path.isfile(fpath + save_name):
            _log("[_download_dataset()] {0} Already Exists. Download Stopped.".format(fname))
            return
    req.urlretrieve("http://yann.lecun.com/exdb/mnist/" + fname, fpath + save_name)
    _log("[_download_dataset()] Done!")

def _load_pkl(fname):
    _log("[_load_pkl()] Loading data from {0}.pkl...".format(fname))
    data = None
    with open("MNIST_pkl/" + fname + ".pkl", "rb") as fp:
        data = pickle.load(fp)
    _log("[_load_pkl()] Done!")
    return data

def _dump_pkl(fname, data):
    _log("[_dump_pkl()] Output data to {0}.pkl...".format(fname))
    with open("MNIST_pkl/" + fname + ".pkl", "wb") as fp:
        pickle.dump(data, fp)
    _log("[_dump_pkl()] Done!")

def _load_imgs(fname):
    data = None
    if os.path.isfile("MNIST_pkl/" + fname + ".pkl"):
        _log("[_load_imgs()] Loading images from {0}.pkl".format(fname))
        data = _load_pkl(fname)
    else:
        _log("[_load_imgs()] Loading images from {0}".format(fname))
        with gzip.open("MNIST_Raw/" + fname, "rb") as fp:
            data = np.frombuffer(fp.read(), np.uint8, offset=16)
        
        # row 차원 위치에 -1을 넣으면, 
        # shape가 (numOfData / column) * column이 됨.
        print(data.shape)
        data = data.reshape(-1, 784)
        _dump_pkl(fname, data)

    _log("[_load_imgs()] Done!")
    return data

def _load_labels(fname):
    data = None
    if os.path.isfile("MNIST_pkl/" + fname + ".pkl"):
        _log("[_load_labels()] Loading labels from {0}".format(fname + ".pkl"))
        data = _load_pkl(fname)
    else:
        _log("[_load_labels()] Loading labels from {0}".format(fname))
        with gzip.open("MNIST_Raw/" + fname, "rb") as fp:
            data = np.frombuffer(fp.read(), np.uint8, offset=8)

        _dump_pkl(fname, data)

    _log("[_load_labels()] Done!")
    return data

def _one_hot_encoder(x):
    tmp = np.zeros((x.size, 10))

    for i in range(x.shape[0]):
        tmp[i][x[i]] = 1
    
    return tmp

def _init_MNIST():
    s = time.time()
    if not os.path.isdir("MNIST_pkl/"):
        os.mkdir("MNIST_pkl/")

    _log("[_init_MNIST()] Downloading MNIST Dataset...")
    _download_dataset("train-images-idx3-ubyte.gz", "train_img.gz")
    _download_dataset("train-labels-idx1-ubyte.gz", "train_lbl.gz")
    _download_dataset("t10k-images-idx3-ubyte.gz", "test_img.gz")
    _download_dataset("t10k-labels-idx1-ubyte.gz", "test_lbl.gz")
    _log("[init_MNIST()] Dataset Downloaded.")

    _log("[_init_MNIST()] Loading Dataset...")
    train_imgs = _load_imgs("train_img.gz")
    train_lbls = _load_labels("train_lbl.gz")

    test_imgs = _load_imgs("test_img.gz")
    test_lbls = _load_labels("test_lbl.gz")
    _log("[_init_MNIST()] Dataset Loaded.")
    _log("[_init_MNIST()] Elapsed Time: {0}".format(time.time() - s))

    return (train_imgs, train_lbls), (test_imgs, test_lbls)

def load_MNIST(normalize=True, flatten=True, one_hot_encoding=True):
    (train_imgs, train_lbls), (test_imgs, test_lbls) = _init_MNIST()

    if normalize:
        train_imgs = train_imgs / 255
        test_imgs = test_imgs / 255

    if not flatten:
        train_imgs = train_imgs.reshape(-1, 1, 28, 28)
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)

    if one_hot_encoding:
        train_lbls = _one_hot_encoder(train_lbls)
        test_lbls = _one_hot_encoder(test_lbls)

    return (train_imgs, train_lbls), (test_imgs, test_lbls)

if __name__ == "__main__":
    (_, _), (_, _) = load_MNIST()