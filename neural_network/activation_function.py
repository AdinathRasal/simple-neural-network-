import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

def my_softmax(x):
    softmax_ = []
    for i in x[0]:
        s = np.sum(x)
        softmax_.append(i/s)

    return[softmax_]


def D_softmax(x):
    return softmax(x) * (1-softmax(x))

def D_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def D_relu(x):
    """
    Computes the derivative of the ReLU activation function.
    
    Args:
    x (float or numpy array): Input value(s) to the ReLU function.
    
    Returns:
    float or numpy array: Derivative of the ReLU function at the input value(s).
    """
    if isinstance(x, (int, float)):
        return 1 if x > 0 else 0
    else:
        return (x > 0).astype(int)


def D_tanh(x):
    """
    Compute the derivative of the hyperbolic tangent (tanh) function.
    
    Args:
    x: numpy array, input to the tanh function
    
    Returns:
    numpy array, derivative of the tanh function
    """
    return 1 - np.tanh(x)**2


def jugad_sum(x):
    x_shape = np.array(x).shape
    product = np.zeros((len(x[0])))
    for index , i in enumerate(x):

        for j in range(len(i)):

            if index < len(x)-2*index:
                
                product[j] += x[2*index][j] + x[(2*index)+1][j]

        if index == (len(x)-1) and len(x)%2 == 1:
            product += x[-1]

    return np.reshape(product,x_shape)
