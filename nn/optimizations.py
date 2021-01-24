import numpy as np

class GradientDescent:
    '''Gradient descent optimizer.
    '''
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @staticmethod
    def initialize(parameters) :
        pass


    def update_parameters(self, parameters, grads):
        '''Updates parameters using gradient descent.

        Args:
            parameters: dictionary of the parameters.
            grads: dictionary contains the gradients of the parameters.
        Returns:
            dictionary of the updated parameters.
        '''
        num_layers = len(parameters) // 2 


        for l in range(1, num_layers+1):
            parameters[f"W{l}"] = parameters[f"W{l}"]-self.learning_rate*grads[f"dW{l}"]
            parameters[f"b{l}"] = parameters[f"b{l}"]-self.learning_rate*grads[f"db{l}"]

        return parameters



class Momentum:
    '''Gradient descent (with momentum) optimizer.
    '''
    def __init__(self, learning_rate, beta = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta

        self.v = {}


    def initialize(self, parameters):
        '''Initializes the velocity.

        Args:
            parameters: dictionary of the parameters.
        '''
        
        L = len(parameters) // 2
        self.v = {}
        
        for l in range(L):
            self.v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])
            self.v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])
            


    def update_parameters(self, parameters, grads):
        '''Updates parameters using gradient descent (with momentum).

        Args:
            parameters: dictionary of the parameters.
            grads: dictionary contains the gradients of the parameters.
        Returns:
            dictionary of the updated parameters.
        '''

        L = len(parameters) // 2 
        

        for l in range(L):
            
            self.v["dW" + str(l + 1)] = self.beta * self.v["dW" + str(l + 1)] + (1 - self.beta) * grads['dW' + str(l + 1)]
            self.v["db" + str(l + 1)] = self.beta * self.v["db" + str(l + 1)] + (1 - self.beta) * grads['db' + str(l + 1)]

            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * self.v["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * self.v["db" + str(l + 1)]
            
        return parameters





class Adam:
    '''Optimizer that implements the Adam algorithm.
    '''
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.v = {}
        self.s = {}

    def initialize(self, parameters) :
        '''Initializes the v and s for Adam optimizer.

        Args:
            parameters: dictionary of the parameters.
        '''
        
        L = len(parameters) // 2 
        self.t = 0
        
        for l in range(L):
            self.v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
            self.v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

            self.s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
            self.s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
        


    def update_parameters(self, parameters, grads):
        '''Updates parameters using Adam.

        Args:
            parameters: dictionary of the parameters.
            grads: dictionary contains the gradients of the parameters.
        Returns:
            dictionary of the updated parameters.
        '''

        L = len(parameters) // 2                 
        v_corrected = {}                       
        s_corrected = {}                         
        self.t += 1


        for l in range(L):

            self.v["dW" + str(l + 1)] = self.beta1 * self.v["dW" + str(l + 1)] + (1 - self.beta1) * grads['dW' + str(l + 1)]
            self.v["db" + str(l + 1)] = self.beta1 * self.v["db" + str(l + 1)] + (1 - self.beta1) * grads['db' + str(l + 1)]

            v_corrected["dW" + str(l + 1)] = self.v["dW" + str(l + 1)] / (1 - np.power(self.beta1, self.t))
            v_corrected["db" + str(l + 1)] = self.v["db" + str(l + 1)] / (1 - np.power(self.beta1, self.t))


            self.s["dW" + str(l + 1)] = self.beta2 * self.s["dW" + str(l + 1)] + (1 - self.beta2) * np.power(grads['dW' + str(l + 1)], 2)
            self.s["db" + str(l + 1)] = self.beta2 * self.s["db" + str(l + 1)] + (1 - self.beta2) * np.power(grads['db' + str(l + 1)], 2)


            s_corrected["dW" + str(l + 1)] = self.s["dW" + str(l + 1)] / (1 - np.power(self.beta2, self.t))
            s_corrected["db" + str(l + 1)] = self.s["db" + str(l + 1)] / (1 - np.power(self.beta2, self.t))


            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + self.epsilon)
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + self.epsilon)

        return parameters