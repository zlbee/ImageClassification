import os
import pickle
import numpy as np
import load_STL10

# CIFAR 10
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# https://stackoverflow.com/questions/37512290/reading-cifar10-dataset-in-batches
def load_pickle(f):
    return pickle.load(f, encoding='latin1')


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def decode_imgs(img_data):
    # Reshape the whole image data
    img_data = img_data.reshape(len(img_data), 3, 32, 32)
    # Transpose the whole data
    img_data = img_data.transpose(0, 2, 3, 1)
    return img_data


# STL 10
def read_STL10():
    x_train = load_STL10.read_all_images('../data/stl10_binary/train_X.bin')
    y_train = load_STL10.read_labels('../data/stl10_binary/train_y.bin')
    x_test = load_STL10.read_all_images('../data/stl10_binary/test_X.bin')
    y_test = load_STL10.read_labels('../data/stl10_binary/test_y.bin')
    return x_train, y_train, x_test, y_test
