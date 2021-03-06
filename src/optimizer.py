import numpy as np
import time

"""
General class of Optimizer
"""
class Optimizer:

    def __init__(self, params, loss, learn_rate, tollerance):
        """
        Parameters:
        params: array of 'weights' used as variable in the optimization with shape (number_features, 1)
        loss: loss function to minimize (must be Loss Object from loss.py script)
        learn_rate: the learning rate (float)
        """
        self.params = params
        self.learn_rate = learn_rate
        self.loss = loss
        self.tollerance = tollerance

    """
    Step rule
    """
    def step(self, X, y, verbose=0):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        verbose: boolean indicating if display step information. Default is 0
        """

        raise NotImplementedError

    """
    Main method of Optimizer. Run n iteration of step
    """
    def run(self, X, y, epoch, verbose=0):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        epoch: integer indicating how many maximum step are to be done
        verbose: boolean indicating if display step information. Default is 0
        """

        raise NotImplementedError


"""
Classic Gradient Descent method with fixed step size.
To see description of initial parameters and methods see Optimizer
"""
class GD(Optimizer):

    def step(self, X, y):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        """
        ### 1. Compute loss and gradient
        crnt_loss = self.loss.compute_loss(X, y, self.params)
        crnt_gradient = self.loss.compute_gradient(X, y, self.params)
        # print('Current gradient shape is: {}'.format(crnt_gradient.shape))
        ### 2. Update phase
        self.params = self.params - self.learn_rate * crnt_gradient
        return crnt_loss, crnt_gradient

    def run(self, X, y, num_epochs, verbose=0):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        epoch: integer indicating how many maximum step are to be done
        verbose: boolean indicating if display step information. Default is 0
        """
        ### 1. Init list where to save results
        loss_list, params_list, time_list = [], [], []
        ### 2. Main iteration of step
        start_time = time.time()
        for n in range(num_epochs):
            ### 2.1 Call step to return loss and gradient and update parameters
            loss, gradient = self.step(X, y)
            ### 2.2 Save current loss and parameters
            loss_list.append(loss)
            params_list.append(self.params)
            time_list.append(time.time() - start_time)
            ### 2.3 Display info if verbose option is True
            if verbose:
                print('Step number {}:'.format(n+1))
                print('\tCurrent loss is: {}'.format(loss))
                print('\tCurrent gradient is: {}'.format(gradient))
                print('\tCurrent parameters are: {}'.format(self.params))
                print('\tElapsed time is: {}'.format(time_list[-1]))
            ### Stopping criterion
            if np.linalg.norm(gradient) < self.tollerance:
                break
        results = {'loss_list': loss_list, 'params_list': params_list, 'time_list': time_list}
        return results


class SGD(GD):

    def run(self, X, y, num_epochs, verbose=0):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        epoch: integer indicating how many maximum step are to be done
        verbose: boolean indicating if display step information. Default is 0
        """
        # Initialize loss, weights and time lists
        loss_list, params_list, time_list = [], [], []
        # Get number of rows
        m = X.shape[0]
        # Iterate through each required epoch
        start_time = time.time()
        for n in range(num_epochs):
            # Choose a random realization among the input dataset
            for i in np.random.choice(m, m, replace=True):
                # Update parameters, get loss and gradient
                loss, gradient = self.step(X[i, :].reshape(1, -1), y[i, :])
                # Store loss, gradient and time for current step
                loss_list.append(loss)
                params_list.append(self.params)
                time_list.append(time.time() - start_time)
                # Verbose output
                if verbose:
                    print('Step number {}:'.format(i+i*n+1))
                    print('\tCurrent loss is: {}'.format(loss))
                    print('\tCurrent gradient is: {}'.format(gradient))
                    print('\tCurrent parameters are: {}'.format(self.params))
                    print('\tElapsed time is: {}'.format(time_list[-1]))
                ### Stopping criterion
                if np.linalg.norm(gradient) < self.tollerance:
                    break
        results = {'loss_list': loss_list, 'params_list': params_list, 'time_list': time_list}
        return results


class SVRG(Optimizer):

    # Constructor
    def __init__(self, params, loss, learn_rate, tollerance, iter_epoch, prev_params=None):
        # Call parent constructor
        super().__init__(params, loss, learn_rate, tollerance)
        # Set previous parameters
        self.prev_params = params if not prev_params else prev_params
        self.iter_epoch = iter_epoch

    def step(self, X, y, prev_gradients):
        # Initialize m and gamma
        m = X.shape[0]
        gamma = 1
        # Get random row index in g
        ik = np.random.choice(m)
        # Compute loss and gradient for i-th input vector
        loss = self.loss.compute_loss(X[ik, :].reshape(1, -1), y[ik, :], self.params)
        gradient = self.loss.compute_gradient(X[ik, :].reshape(1, -1), y[ik, :], self.params)
        # Compute gradient for ik-th observation using previous parameters
        prev_gradient_ik = self.loss.compute_gradient(X[ik, :].reshape(1, -1), y[ik, :], self.prev_params)
        # Then add leftmost side of the formula
        update = gamma * (gradient - prev_gradient_ik) + prev_gradients
        # Update parameters with gradient
        self.params = self.params - self.learn_rate * update
        # Return either loss and gradient
        return loss, gradient

    def run(self, X, y, num_epochs, verbose=0):
        # Initialize lists of either loss and gradient
        loss_list, params_list, time_list = [], [], []
        # Loop through each epoch
        start_time = time.time()
        for n in range(num_epochs):
            # Compute update vector: start by rightmost part of the formula
            prev_gradients = self.loss.compute_gradient(X, y, self.prev_params)
            # Loop through each observation in epoch
            for l in range(self.iter_epoch):
                # Compute update step for every epoch
                loss, gradient = self.step(X, y, prev_gradients)
                # Store loss, gradient and time for current step
                loss_list.append(loss)
                params_list.append(self.params)
                time_list.append(time.time() - start_time)
                # Verbose output
                if verbose:
                    print('Step number {}:'.format(l+l*n+1))
                    print('\tCurrent loss is: {}'.format(loss))
                    print('\tCurrent gradient is: {}'.format(gradient))
                    print('\tCurrent parameters are: {}'.format(self.params))
                    print('\tElapsed time is: {}'.format(time_list[-1]))
            # Update stored parameters
            self.prev_params = self.params.copy()
            ### Stopping criterion
            if np.linalg.norm(gradient) < self.tollerance:
                break
        results = {'loss_list': loss_list, 'params_list': params_list, 'time_list': time_list}
        return results
