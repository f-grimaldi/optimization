"""
General class of Optimizer
"""
class Optimizer:

    def __init__(self, params, loss, learn_rate):
        """
        Parameters:
        params: array of 'weights' used as variable in the optimization with shape (number_features, 1)
        loss: loss function to minimize (must be Loss Object from loss.py script)
        learn_rate: the learning rate (float)
        """
        self.params = params
        self.learn_rate = learn_rate
        self.loss = loss

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

    def step(self, X, y, verbose=0):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        verbose: boolean indicating if display step information. Default is 0
        """

        ### 1. Compute loss and gradient
        crnt_loss = self.loss.compute_loss(X, y, self.params)
        crnt_gradient = self.loss.compute_gradient(X, y, self.params)
        ### 2. Update phase
        self.params = self.params - self.learn_rate*crnt_gradient
        return crnt_loss, crnt_gradient

    def run(self, X, y, epoch, verbose=0):
        """
        Parameters:
        X: matrix of input with shape (number_inputs, number_features)
        y: array of target with shape (number_inputs, 1)
        epoch: integer indicating how many maximum step are to be done
        verbose: boolean indicating if display step information. Default is 0
        """

        ### 1. Init list where to save results
        loss_list = []
        params_list = []
        ### 2. Main iteration of step
        for n in range(epoch):
            ### 2.1 Call step to return loss and gradient and update parameters
            loss, gradient = self.step(X, y, verbose)
            ### 2.2 Save current loss and parameters
            loss_list.append(loss)
            params_list.append(self.params)
            ### 2.3 Display info if verbose option is True
            if verbose:
                print('Step number {}:'.format(n+1))
                print('\tCurrent loss is: {}'.format(loss))
                print('\tCurrent gradient is: {}'.format(gradient))
                print('\tCurrent parameters are: {}'.format(self.params))
        """
        TODO: Stop criterion
        """
        return {'loss_list': loss_list, 'params_list': params_list}


class SGD(Optimizer):

    def step(self, X, y):
        raise NotImplementedError

    def run(self, X, y):
        raise NotImplementedError


class SVRG(Optimizer):

    def step(self, X, y):
        raise NotImplementedError

    def run(self, X, y):
        raise NotImplementedError
