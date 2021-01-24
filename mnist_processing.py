import numpy as np
import os
import matplotlib.pyplot as plt
import gzip


data_path = 'D:/Content_2/NN from scratch/NeuralNetworks/data/MNIST'


def load_images(path):
    """Return images loaded locally."""
    with gzip.open(path) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784)

def load_labels(path):
    """Return labels loaded locally."""
    with gzip.open(path) as f:
        # First 8 bytes are magic_number, n_labels
        integer_labels = np.frombuffer(f.read(), 'B', offset=8)
    return integer_labels



def load_mnist(data_path):
    X_train = load_images(data_path + '/train-images-idx3-ubyte.gz')
    y_train = load_labels(data_path + '/train-labels-idx1-ubyte.gz')

    X_test = load_images(data_path + '/t10k-images-idx3-ubyte.gz')
    y_test = load_labels(data_path + '/t10k-labels-idx1-ubyte.gz')
    return X_train, y_train, X_test, y_test



def show_img(img):
    plt.imshow(img.reshape(28, 28))
