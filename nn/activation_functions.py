import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def softmax(Z):
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=0, keepdims=True)

def get_activation(Z, name):
    if name == 'sigmoid':
        return sigmoid(Z)
    elif name == 'relu':
        return relu(Z)
    elif name == 'softmax':
        return softmax(Z)

def grad_relu(dA, Z):
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def grad_sigmoid(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    return dZ

def grad_softmax(dA, Z):
    return dA


def get_grad_activation(dA, Z, name):
    if name == 'sigmoid':
        return grad_sigmoid(dA, Z)
    elif name == 'relu':
        return grad_relu(dA, Z)
    elif name == 'softmax':
        return grad_softmax(dA, Z)