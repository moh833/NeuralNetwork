import numpy as np

def binary_cross_entropy(AL, Y):
    '''Implements binary-cross-entropy cost.

    Args:
        AL: probability vector of label predictions.
        Y: true labels vector.
    Returns:
        the total binary-cross-entropy cost.
    '''
    m = Y.shape[1]

    logprods = np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T)
    total_cost = -np.sum(logprods)
    
    total_cost = np.squeeze(total_cost)      
    assert(total_cost.shape == ())
    
    return total_cost

def grad_binary_cross_entropy(AL, Y):
    '''Implements the derivative of binary-cross-entropy cost.

    Args:
        AL: probability vector of label predictions.
        Y: true labels vector.
    Returns:
        the derivative of binary-cross-entropy cost.
    '''
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    return dAL


def categorical_cross_entropy(AL, Y):
    '''Implements categorical-cross-entropy cost.

    Args:
        AL: probability vector of label predictions.
        Y: true labels vector.
    Returns:
        the total categorical-cross-entropy cost.
    '''
    total_cost = np.sum(-np.log( AL[ Y, np.arange(Y.shape[1]) ] )) 

    total_cost = np.squeeze(total_cost)      
    assert(total_cost.shape == ())

    return total_cost

def grad_categorical_cross_entropy(AL, Y):
    '''Implements the derivative of categorical-cross-entropy cost.

    Args:
        AL: probability vector of label predictions.
        Y: true labels vector.
    Returns:
        the derivative of categorical-cross-entropy cost.
    '''
    dAL = np.copy(AL)
    dAL[Y, np.arange(Y.shape[1])] -= 1

    return dAL 


def compute_cost(AL, Y, name):
    '''Computes the cost of a specified type.

    Args:
        AL: probability vector of label predictions.
        Y: true labels vector.
        name: type of the cost to compute.
    Returns:
        the cost of the specified type.
    '''
    if name =='binary-cross-entropy':
        return binary_cross_entropy(AL, Y)
    elif name == 'categorical-cross-entropy':
        return categorical_cross_entropy(AL, Y)


def grad_cost(AL, Y, name):
    '''Computes the derivative of a specified cost.

    Args:
        AL: probability vector of label predictions.
        Y: true labels vector.
        name: type of the cost to compute it's derivative.
    Returns:
        the derivative of the specified cost.
    '''
    if name =='binary-cross-entropy':
        return grad_binary_cross_entropy(AL, Y)
    elif name == 'categorical-cross-entropy':
        return grad_categorical_cross_entropy(AL, Y)


def compute_reg_cost(parameters, L2_regularization_lambdas):
    '''Computes the L2 regularization cost.

    Args:
        parameters: dictionary of the parameters.
        L2_regularization_lambdas: list of the lambdas for each layer, None if no reg is applied to this layer.
    Returns:
        the total regularization cost.
    '''
    reg_cost = 0
    num_layers = len(parameters) // 2
    
    for l in range(1, num_layers+1):
        if L2_regularization_lambdas[l-1]:
            reg_cost += L2_regularization_lambdas[l-1] * np.sum( np.square( parameters[f'W{l}'] ) )
    
    return reg_cost


def mean_squared_error(AL, Y):
    total_cost = np.sum(np.square(AL - Y))

    total_cost = np.squeeze(total_cost)      
    assert(total_cost.shape == ())

    return total_cost


def mean_squared_error_backward(AL, Y):
    dAL = 1/2 * np.mean( (AL - Y) , axis=1)
    return dAL



