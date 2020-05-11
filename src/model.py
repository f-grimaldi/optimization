import numpy as np

"""
Linear Classifier
"""
class Model:

    def __init__(self, input_size, init_weights='random'):
        """
        Parameters:
        input_size:     the number of features. Int
        init_weights:   the type of  weights iniztialization. It can be random or zeros.
        """
        self.input_size = input_size
        self.init_weights(init_weights)

    """
    Fit method
    """
    def fit(self, X, y, optimizer, num_epochs, verbose = 0):
        """
        Parameters:
        X:             Matrix of input with shape (number_of_examples, number_of_features). np.matrix
        y:             Array/Matrix of target with shape (number_of_examples, 1). np.matrix
        optimizer:     The optimzer choosen from the available in optimizer.py. Optimizer Object
        num_epochs:    The number of steps of training. Int
        verbose:       Display step information. Bool. Default is 0
        """
        ### 1. Add the bias unit to the input
        X = self.add_bias(X)
        ### 2. Run the search for the minima with the chosen optimizer
        results = optimizer.run(X, y, num_epochs, verbose)
        ### 3. Update the weights with the last result given by the optimizer
        self.weights = results['params_list'][-1]
        ### 4. Get the best treshold
        #self.threshold = self.get_threshold(X, y)
        ### 4. Return the losses
        return results

    """
    Predict the output of the model with a given input
    """
    def predict(self, X):
        """
        Parameters:
        X:             Matrix of input with shape (number_of_examples, number_of_features). np.matrix
        """
        # 1. Add bias
        X = self.add_bias(X)
        # 2. Return output
        try:
            return np.dot(X, self.weights) - self.threshold
        except:
            return np.dot(X, self.weights)

    """
    Add a bias/slope to the input
    """
    def add_bias(self, X):
        """
        Parameters:
        X:             Matrix of input with shape (number_of_examples, number_of_features). np.matrix
        """
        if X.shape[1] == self.weights.shape[0]:
            return X
        ### 1. Create a matrix with same row of X and 1 column more
        mat = np.zeros((X.shape[0], X.shape[1]+1))
        ### 2. Fill the matrix with the example with the bias feature (1)
        mat[:, :-1] = X
        mat[:, -1] = 1
        return mat

    """
    To implement if the threshold of the ooutput is neither 0.5 or 0
    """
    def get_threshold(self, X, y):
        """
        X:             Matrix of input with shape (number_of_examples, number_of_features). np.matrix
        y:             Array/Matrix of target with shape (number_of_examples, 1). np.matrix
        """
        ### 1. Get the prediction
        y_pred = self.predict(X)
        ### 2. Compute threshold
        positive = y_pred[y == 1]
        negative = y_pred[y == -1]
        return np.mean([np.min(positive), np.max(negative)])

    """
    Various iniztialization method for the model weights
    """
    def init_weights(self, init_type):
        ### 1. Initalize the weghts
        if init_type == 'random':
            self.weights = np.random.randn(self.input_size+1, 1)
        elif init_type == 'zeros':
            self.weights = np.zeros((self.input_size+1, 1))
        else:
            raise NotImplementedError


if __name__ == "__main__":

    import optimizer
    import loss

    import argparse
    import time
    from tqdm import tqdm

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score


    X = np.random.randint(-10, 10, (30, 1))
    y = np.random.randint( -1,  2, (30, 1))
    y[X <  0] = -1
    y[X >= 0] =  1

    print('Data:')
    print('\tInput: {}'.format(X.transpose()))
    print('\tTarget: {}'.format(y.transpose()))

    loss_fn = loss.LogisticLoss(reg_coeff = 0.02)
    W = np.arange(-5, 10, 0.2)
    losses = [np.round(loss_fn.compute_loss(X, y, np.array([[w]])), 3) for w in W]

    optim = optimizer.GD(params=np.array([[0]]), loss=loss_fn, learn_rate=0.02)
    results = optim.run(X, y, 20)

    print('\nComputed losses:\n\t{}\n'.format(losses))
    print('Results:')
    print('\tLosses: {}'.format(np.round(results['loss_list'], 3)))
    print('\tParams: {}'.format([np.round(float(i[0]), 3) for i in results['params_list']]))
    print('\tOutput: {}'.format(np.dot(X, results['params_list'][-1]).transpose()))


    plt.plot(W, losses, label='Logistic Loss Function')
    plt.scatter(results['params_list'][:-1], results['loss_list'][1:], c='r', s=15, label = 'Optimizer step')
    plt.xlabel('W')
    plt.ylabel('l(W; (X,y))')
    plt.title('1D dimension example')
    plt.legend()
    plt.grid()
    plt.show()


    ### Create input and output
    X = np.random.randint(-5, 5, (100, 1))
    y = np.random.randint(-1, 2, (100, 1))
    y[X <  2] = -1
    y[X >= 2] =  1
    mat = np.zeros((X.shape[0], X.shape[1]+1))
    mat[:, :-1] = X
    mat[:,  -1] = 1
    X = mat

    loss_fn = loss.LogisticLoss(reg_coeff = 0.1)
    optim = optimizer.GD(params=np.random.randn(2, 1), loss=loss_fn, learn_rate=0.3)
    results = optim.run(X, y, 30)
    p1, p2 = [], []
    for i in results['params_list']:
        p1.append(i[0])
        p2.append(i[1])


    w1, w2 = np.arange(-3, 5, 0.1), np.arange(-3, 5, 0.1)
    losses = np.zeros((w1.shape[0], w2.shape[0]))
    for i in tqdm(range(w1.shape[0])):
        for j in range(w2.shape[0]):
            l = loss_fn.compute_loss(X, y, np.array([w1[i], w2[j]]))
            losses[i, j] = l

    W1, W2 = np.meshgrid(w1,w2)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot a 3D surface
    ax.set_title('2D example (with bias)')
    ax.plot_wireframe(W1, W2, losses, rstride=5, cstride=5)
    ax.scatter(p1[:-1], p2[:-1], results['loss_list'][1:], color="r", s=15)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    plt.show()

    print('Loss: ', results['loss_list'][-1])
    print('Params: ', results['params_list'][-1])
    print('Computed Loss: ', loss_fn.compute_loss(X, y, np.array(results['params_list'][-1])))
    print(results['loss_list'])
