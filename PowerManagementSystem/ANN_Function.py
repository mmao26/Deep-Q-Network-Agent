"""
Created on March 17 2018

@author: Manqing Mao
@modified by: Manqing Mao
@description: v2.0 of supervised learning for ANN in Python

"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
np.random.seed(0) # TODO: CHeck this random seed should not change our PRNB sequence 

class Config:
    nn_input_dim = 9  # input layer dimensionality
    nn_output_dim = 3  # output layer dimensionality
    nn_hdim0 = 9
    nn_hdim1 = 4
    # Gradient descent parameters (I picked these by hand)
    learning_rate = 0.1 #0.1#0.001  # learning rate for gradient descent
    reg_lambda = 0#.001  # L2-regularization strength
    
    classify = True # False for DQN, True for classification.
    RELU = 1

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y, RELU=Config.RELU, classify=Config.classify):
    num_examples = len(X)  # training set size
    W0, b0, W1, b1, W2, b2 = model['W0'], model['b0'], model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z0 = X.dot(W0) + b0
    a0 = leakyRelu(z0) if RELU else np.tanh(z0)
    z1 = a0.dot(W1) + b1
    a1 = leakyRelu(z1) if RELU else np.tanh(z1)
    z2 = a1.dot(W2) + b2

    # Calculating the loss
    if classify:
        # softmax for classification
        # cross-entropy loss
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
    else:
        # mse loss function for DQN
        data_loss = np.square(np.subtract(z2, y)).mean()
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W0)) + np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def predict(model, x, RELU=Config.RELU, classify=Config.classify):
    W0, b0, W1, b1, W2, b2 = model['W0'], model['b0'], model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z0 = x.dot(W0) + b0
    a0 = leakyRelu(z0) if RELU else np.tanh(z0)
    z1 = a0.dot(W1) + b1
    a1 = leakyRelu(z1) if RELU else np.tanh(z1)
    z2 = a1.dot(W2) + b2

    if classify:
        # softmax for classification
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
    else:
        # linear for DQN
        return z2

# Leaky Relu Function [Debugged]
def leakyRelu (x):
    return np.maximum(x, 0.01*x)

# Derivative of Leaky Relu Function [Debugged]
def derivatives_leakyRelu(x):
    x[x<0] = 0.01
    #x[x==0] = (1.0 + 0.01)/2.0
    x[x>=0] = 1.0
    return x

def build_model(): #[Debugged]
    
    W0 = np.random.randn(Config.nn_input_dim, Config.nn_hdim0) / np.sqrt(Config.nn_input_dim/2.0) # Normalize the weights to start with
    b0 = np.zeros((1, Config.nn_hdim0))
    W1 = np.random.randn(Config.nn_hdim0, Config.nn_hdim1) / np.sqrt(Config.nn_hdim0/2.0)
    b1 = np.zeros((1, Config.nn_hdim1))
    W2 = np.random.randn(Config.nn_hdim1, Config.nn_output_dim) / np.sqrt(Config.nn_hdim1/2.0)
    b2 = np.zeros((1, Config.nn_output_dim))

    # This is what we return at the end
    model = {}
    # Assign new parameters to the model [Store the data in Python dictionary]
    model = {'W0': W0, 'b0': b0, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model
    
# This function learns parameters for the neural network and returns the model.
# - nn_hdim0: Number of nodes in the hidden layer0
# - epoch: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations.
# - batch_size: minibatch, granularity of weights update
# - RELU: choose activation of leakyrelu if RELU=1, otherwise choose of tanh
#[Debugged]   
def fit_model(X, y, init_model, epoch, batch_size=1, RELU=Config.RELU, print_loss=False, classify=Config.classify):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    num_batch = int(np.ceil(num_examples/batch_size))
    
    W0, b0, W1, b1, W2, b2 = init_model['W0'], init_model['b0'], init_model['W1'], init_model['b1'], init_model['W2'], init_model['b2']
    
    # Gradient descent. For each epoch...
    for i in range(0, epoch):
        for m in range(0, num_batch):

            start_idx = m*batch_size
            end_idx = min((m+1)*batch_size, num_examples)
            X_batch = X[start_idx:end_idx]
            #print ("X_batch is ")
            #print (X_batch)

            y_batch = y[start_idx:end_idx]
            #print ("y_batch is ")
            #print (y_batch)

            # Forward propagation
            z0 = X_batch.dot(W0) + b0
            a0 = leakyRelu(z0) if RELU else np.tanh(z0)
            z1 = a0.dot(W1) + b1
            a1 = leakyRelu(z1) if RELU else np.tanh(z1)
            z2 = a1.dot(W2) + b2    

            # Backpropagation
            if classify:
                # softmax for classification
                exp_scores = np.exp(z2) # take exp
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # normalize
                delta3 = probs
                delta3[range(len(X_batch)), y_batch] -= 1 # classification
            else:
                # linear for DQN
                delta3 = z2 - y_batch # Error at the output layer for the Q values

            # Update the weights for the network between last hidden layer and output
            dW2 = (a1.T).dot(delta3) # Change in weights with activations of the hidden layer
            db2 = np.sum(delta3, axis=0, keepdims=True) # weight of the bias input
            delta2 = delta3.dot(W2.T) * derivatives_leakyRelu(a1) if RELU else delta3.dot(W2.T) * (1 - np.power(a1, 2))
            
            dW1 = np.dot(a0.T, delta2)
            db1 = np.sum(delta2, axis=0)
            delta1 = delta2.dot(W1.T) * derivatives_leakyRelu(a0) if RELU else delta2.dot(W1.T) * (1 - np.power(a0, 2))
            
            dW0 = np.dot(X_batch.T, delta1)
            db0 = np.sum(delta1, axis=0)

            # Add regularization terms (b0, b1 and b2 don't have regularization terms)
            dW2 += Config.reg_lambda * W2
            dW1 += Config.reg_lambda * W1
            dW0 += Config.reg_lambda * W0
        
            # St. Gradient descent parameter update
            W0 += -Config.learning_rate * dW0
            b0 += -Config.learning_rate * db0
            W1 += -Config.learning_rate * dW1
            b1 += -Config.learning_rate * db1
            W2 += -Config.learning_rate * dW2
            b2 += -Config.learning_rate * db2
            
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(init_model, X, y, classify=Config.classify)))

    return init_model

def main():

    X = np.array([[1,0,1,0,1,0,1,0,1],[1,0,1,1,1,0,1,0,0],[0,1,0,1,1,0,1,0,1],[0,0,0,1,1,1,1,0,0]])
    y = np.array([0,1,2,2])
    
    init_model = build_model()
    model = fit_model(X, y, init_model, epoch=10000, batch_size=3, print_loss=True)
    y_est = predict(model, X)
    #print("Model is", model)
    print("Estimation is", y_est)
    print("\n%s: %.2f%%" % ("Accuracy", 100*np.sum(y_est == y)/len(X)))
    


if __name__ == "__main__":
    main()

