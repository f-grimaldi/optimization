"""
Logistic Regression Class
"""
import numpy as np
import optim

class LogisticRegressor():
    def __init__(self, input_size, bias=True):
        self.dim = input_size
        self.bias = bias
        self.init()
        self.is_fit = False
        self.normalized = False

    """
    Initialize the weights with normal distribution
    """
    def init(self):
        if self.bias:
            self.W = np.random.randn(1, self.dim+1)
        else:
            self.W = np.random.randn(1, self.dim)

    """
    Add bias feature
    """
    def add_bias(self, X):
        if self.bias:
            mat = np.zeros((X.shape[0], X.shape[1]+1))
            mat[:, :-1] = X
            mat[:, -1] = 1
            return mat
        else:
            return X

    """
    Learn weight given an optimizer and a training set (X, y)
    """
    def fit(self, X, y, optimizer, epoch=100, stop_criterion = 0.01, normalized = True, verbose=0):
        ### 1. Save optimizer
        self.optim = optimizer

        ### 2. Add Bias Unit
        X = self.add_bias(X)

        ### 3. Choose proper optimizer
        ### CLASSIC
        if type(self.optim) == optim.GradientDescent:
            loss = []
            for ep in range(epoch):
                ### 3.1 Do a step
                ### 3.1.a Call the optimizer
                tmp_loss, gradient = self.optim.step(X, y)
                loss.append(tmp_loss)
                ### 3.1.b Update parameters
                self.W = self.optim.prm

                ### 3.2 Check stopping condition
                """
                TODO
                """

                ### 3.3 Display current results
                if verbose:
                    print('Step:  {}\t\tLoss: {}'.format(ep, np.round(loss[-1], 5)))

        ### SGD
        elif type(self.optim) == optim.SGD:
            loss = []
            for ep in range(epoch):
                ep_loss = []
                for i in range(X.shape[0]):
                    ### 3.1 Do a step
                    ### 3.1.a Call the optimizer
                    tmp_loss, gradient = self.optim.step(X[i, :], y[i])
                    loss.append(tmp_loss)
                    ### 3.1.b Update parameters
                    self.W = self.optim.prm

                    ### 3.2 Check stopping condition
                    """
                    TODO
                    """

                loss.append(np.mean(ep_loss))
                ### 3.3 Display current results
                if verbose:
                    print('Step:  {}\t\tLoss: {}'.format(ep, np.round(loss[-1], 5)))


        ### SVRD
        elif type(self.optim) == optim.SVRG:
            """
            TODO
            """

        ### NONE
        else:
            print('Please use one of the following optimizer:\nGradientDescent, SGD, SVRD. Returning None')
            return None


        ### 4. Normalize the outpush such that decision threshold is 0
        if normalized:
            self.normalized = True
            self.get_threshold(X, y)

        ### 5. Signal model is fitted
        self.is_fit = True

    """
    Predict the data X as a Linear Classifier (X.t*W)
    """
    def predict(self, X, normalized_output=True):
        ### 1. Check if the model is fitted
        if not self.is_fit:
            print('The model is not fitted. Please run LogisticRegressor.fit(X, y) to fit the model. Returning None')
            return None

        ### 2. Add unit bias and forward
        X = self.add_bias(X)
        out = np.dot(X, self.W.transpose()).reshape(-1)

        ### 3. Normalize output such that threshold is 0
        if normalized_output:
            if not self.normalized:
                print("Need to call normalize=True in LogisticRegressor.fit() in order to use this option. Return raw output")
                return out
            else:
                return out-self.threshold

    """
    Compute the threshold and normalized it to 0
    """
    def get_threshold(self, X, y):
        out = np.dot(X, self.W.transpose()).reshape(-1)
        zeros = out[y == -1]
        ones = out[y == 1]
        self.threshold = (ones.min() + zeros.max())/2


    """
    String method
    """
    def __str__(self):
        return "Logistic Regressor\n\tNumber of input: {}\n\tBias:\t\t {}\n".format(self.dim, self.bias)
