import numpy as np 
import matplotlib.pyplot as plt

'''
Objectives : implement the given kernel : Two layer neural network , i think we need to use this kernel in the smo implementation. 
'''


def two_layer_neural_network_kernel(x, y, b, c):
    '''
    x and y : two vectors 
    b,c : two user defined constants 
    '''
    if len(x) != len(y): 
        raise Exception("invalid vector sizes :(")

    suum = 0
    for k in range(len(x)):
        suum += x[k]*y[k]  

    output = np.tanh(b*sum - c)

    return output













