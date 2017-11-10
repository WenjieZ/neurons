import numpy as np

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def dsigmoid(A):
    B = sigmoid(A)
    return B * (1 - B)

def tanh(A):
    return 2 * sigmoid(2 * A) - 1

def dtanh(A):
    return 4 * dsigmoid(2 * A)

def relu(A):
    return np.maximum(A, 0)

def drelu(A):
    return np.sign(relu(A + 0.00001))
    
    
if __name__ == "__main__":
    assert np.equal(sigmoid(0), 0.5)
    assert np.allclose(sigmoid(np.array([0, np.log(3), -np.log(3)])), np.array([ 0.5, 0.75, 0.25])) 
    assert np.allclose(dsigmoid(np.array([0, 100, -100])), np.array([ 0.25, 0, 0])) 
    assert np.equal(tanh(0), 0)
    assert np.isclose(dtanh(1), 1 - tanh(1)**2)
    assert np.equal(relu(-1), 0)
    assert np.equal(relu(1), 1)
    assert np.array_equal(drelu(np.array([-1, 2])), np.array([0, 1]))
    
    
    
