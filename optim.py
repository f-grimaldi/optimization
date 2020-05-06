"""
Collection of optimizers
"""
import numpy as np


class GradientDescent():
    def __init__(self, parameters, lr=0.01, weight_decay=0.001):
        self.prm = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    """
    Compute the loss given X, y
    """
    def compute_loss(self, X, y):
        ones = np.ones((1, X.shape[0])) ### Shape: 1, number_of_imputs
        activation = np.dot(X, self.prm.transpose()) ### Shape number_of_inputs, 1
        regularization = self.weight_decay*(np.linalg.norm(self.prm)**2)*0.5 ### Shape: scalar

        # Sum [log(1 + exp(-y*XW'))] + 0.5*lambda||W||^2
        loss = np.sum(np.log(ones + np.exp(-y*activation.transpose()))) + regularization ### Shape Scalar
        return loss

    """
    Compute gradient
    """
    def compute_gradient(self, X, y):
        first_term = (X.transpose()*y)
        activation = np.dot(X, self.prm.transpose())
        second_term=(1-1/(1+np.exp(y*activation.transpose())))

        gradient = -np.dot(first_term, second_term.transpose()).transpose() + self.weight_decay*self.prm
        return gradient

    """
    Main optimization step
    """
    def step(self, X, y):
        ### 1. Compute the loss
        loss = self.compute_loss(X, y)
        ### 2. Compute Gradient
        gradient = self.compute_gradient(X, y)
        ### 3. Do classic gradient descent with fixed stepsize
        self.prm = self.prm - self.lr*gradient/np.linalg.norm(gradient)
        ### 4. Return loss and gradient
        return loss, gradient

class SGD():
    """
    TODO: Its a copy and paste of GradientDescent Class. The difference as far i ve understood is that in SGD we just feed
    one example at time. SO the difference in code is only done in the LogisticRegressor.fit method.
    """
    def __init__(self, parameters, lr=0.01, weight_decay=0.001):
        self.prm = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    """
    Compute the loss given X, y
    """
    def compute_loss(self, X, y):
        ones = np.ones((1, X.shape[0])) ### Shape: 1, number_of_imputs
        activation = np.dot(X, self.prm.transpose()) ### Shape number_of_inputs, 1
        regularization = self.weight_decay*(np.linalg.norm(self.prm)**2)*0.5 ### Shape: scalar

        # Sum [log(1 + exp(-y*XW'))] + 0.5*lambda||W||^2
        loss = np.sum(np.log(ones + np.exp(-y*activation.transpose()))) + regularization ### Shape Scalar
        return loss

    """
    Compute gradient
    """
    def compute_gradient(self, X, y):
        first_term = (X.transpose()*y)
        activation = np.dot(X, self.prm.transpose())
        second_term=(1-1/(1+np.exp(y*activation.transpose())))

        gradient = -first_term.transpose()*second_term + self.weight_decay*self.prm
        return gradient

    """
    Main optimization step
    """
    def step(self, X, y):
        ### 1. Compute the loss
        loss = self.compute_loss(X, y)
        ### 2. Compute Gradient
        gradient = self.compute_gradient(X, y)
        ### 3. Do classic gradient descent with fixed stepsize
        self.prm = self.prm - self.lr*gradient/np.linalg.norm(gradient)
        ### 4. Return loss and gradient
        return loss, gradient

class SVRD():
    """
    TODO
    """
