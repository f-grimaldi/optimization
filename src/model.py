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
    from matplotlib import cm

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score


    X = np.random.randint(-10, 10, (20, 1))
    y = np.random.randint( -1,  2, (20, 1))
    y[X <  0] = -1
    y[X >= 0] =  1

    # print('Data:')
    # print('\tInput: {}'.format(X.transpose()))
    # print('\tTarget: {}'.format(y.transpose()))

    print('Creating 2d plot...')
    reg_coeff = 0.3
    lims = (-7, 3, 0.2)
    lr = 0.05
    epochs = 40
    starting_point = sp = -5

    loss_fn = loss.LogisticLoss(reg_coeff = reg_coeff)
    W = np.arange(lims[0], lims[1], lims[2])
    losses = [np.round(loss_fn.compute_loss(X, y, np.array([[w]])), 3) for w in W]

    gd  = optimizer.GD(params=np.array([[sp]]), loss=loss_fn, learn_rate=lr)
    sgd = optimizer.SGD(params=np.array([[sp]]), loss=loss_fn, learn_rate=lr)
    sag = optimizer.SAG(params = np.array([[sp]]), loss=loss_fn, learn_rate=lr)
    svrg = optimizer.SVRG(params = np.array([[sp]]), loss=loss_fn, learn_rate=lr, iter_epoch=10)
    results1 = gd.run(X, y, epochs)
    results2 = sgd.run(X, y, epochs//3)
    results3 = sag.run(X,y, epochs//2)
    results4 = svrg.run(X, y, epochs//2)
    # print('\nComputed losses:\n\t{}\n'.format(losses))
    # print('Results:')
    # print('\tLosses: {}'.format(np.round(results['loss_list'], 3)))
    # print('\tParams: {}'.format([np.round(float(i[0]), 3) for i in results['params_list']]))
    # print('\tOutput: {}'.format(np.dot(X, results['params_list'][-1]).transpose()))


    plt.plot(W, losses, label='Logistic Loss Function.\nLambda = {}'.format(reg_coeff))

    plt.scatter(results1['params_list'][:-1], results1['loss_list'][1:], c='r', s=15, label = 'Classic')
    plt.plot([float(i) for i in results1['params_list'][:-1]], results1['loss_list'][1:], c='r')

    plt.scatter(results2['params_list'][:-1], results2['loss_list'][1:], c='green', s=15, label = 'SGD')
    plt.plot([float(i) for i in results2['params_list'][:-1]], results2['loss_list'][1:], c='green')

    plt.scatter(results3['params_list'][:-1], results3['loss_list'][1:], c='yellow', s=15, label = 'SAG')
    plt.plot([float(i) for i in results3['params_list'][:-1]], results3['loss_list'][1:], c='yellow')

    plt.scatter(results4['params_list'][:-1], results4['loss_list'][1:], c='pink', s=15, label = 'SVRG')
    plt.plot([float(i) for i in results4['params_list'][:-1]], results4['loss_list'][1:], c='pink')

    plt.axvline(x=np.argmin(losses)*lims[2]+lims[0], c='black', label = 'Minimum')
    plt.xlabel('W')
    plt.ylabel('l(W; (X,y))')
    plt.title('2D dimension example')
    plt.legend()
    plt.grid()
    plt.show()


    print('Creating 3d plot...')
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


    gd  = optimizer.GD(params =  np.array([-2, -2]).reshape(2, 1), loss=loss_fn, learn_rate=lr)
    sgd = optimizer.SGD(params = np.array([-2, -2]).reshape(2, 1), loss=loss_fn, learn_rate=lr)
    sag = optimizer.SAG(params = np.array([-2, -2]).reshape(2, 1), loss=loss_fn, learn_rate=lr)
    svrg = optimizer.SVRG(params = np.array([-2, -2]).reshape(2, 1), loss=loss_fn, learn_rate=lr, iter_epoch=10)
    results1 = gd.run(X, y, epochs)
    results2 = sgd.run(X, y, epochs//3)
    results3 = sag.run(X,y, epochs//2)
    results4 = svrg.run(X, y, epochs//2)


    p11, p12 = [], []
    p21, p22 = [], []
    p31, p32 = [], []
    p41, p42 = [], []
    for i in results1['params_list']:
        p11.append(float(i[0]))
        p12.append(float(i[1]))

    for j in results2['params_list']:
        p21.append(float(j[0]))
        p22.append(float(j[1]))

    for i in results3['params_list']:
        p31.append(float(i[0]))
        p32.append(float(i[1]))

    for j in results4['params_list']:
        p41.append(float(j[0]))
        p42.append(float(j[1]))

    w1, w2 = np.arange(-3, 5, 0.1), np.arange(-3, 5, 0.1)
    losses = np.zeros((w1.shape[0], w2.shape[0]))
    for i in range(w1.shape[0]):
        for j in range(w2.shape[0]):
            l = loss_fn.compute_loss(X, y, np.array([w1[i], w2[j]]))
            losses[j, i] = l

    W1, W2 = np.meshgrid(w1,w2)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot a 3D surface
    ax.set_title('2D example (with bias)')
    ax.plot_wireframe(W1, W2, losses, rstride=5, cstride=5, label='Logistic Loss Surface')
    cset = ax.contourf(W1, W2, losses, zdir='z', offset=np.min(losses), cmap=cm.ocean)
    #cset = ax.contourf(W1, W2, losses, zdir='x', offset=-5, cmap=cm.ocean)
    #cset = ax.contourf(W1, W2, losses, zdir='y', offset=5, cmap=cm.ocean)
    ax.scatter(p11[:-1], p12[:-1], results1['loss_list'][1:],
               color="r", s=15, label='Classic')
    ax.scatter(p21[:-1], p22[:-1], results2['loss_list'][1:],
               color="orange", s=15, label='SGD')
    ax.scatter(p21[:-1], p22[:-1], results2['loss_list'][1:],
                color="yellow", s=15, label='SAG')
    ax.scatter(p21[:-1], p22[:-1], results2['loss_list'][1:],
                color="pink", s=15, label='SVRG')
    ax.legend(loc=(0.1,0.75),frameon=0)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Loss')
    plt.show()
