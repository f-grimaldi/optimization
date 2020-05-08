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
    def fit(self, X, y, optimizer, epoch, verbose = 0):
        """
        Parameters:
        X:             Matrix of input with shape (number_of_examples, number_of_features). np.matrix
        y:             Array/Matrix of target with shape (number_of_examples, 1). np.matrix
        optimizer:     The optimzer choosen from the available in optimizer.py. Optimizer Object
        epoch:         The number of steps of training. Int
        verbose:       Display step information. Bool. Default is 0
        """
        ### 1. Add the bias unit to the input
        X = self.add_bias(X)
        ### 2. Run the search for the minima with the chosen optimizer
        results = optimizer.run(X, y, epoch, verbose)
        ### 3. Update the weights with the last result given by the optimizer
        self.weights = results['params_list'][-1]
        ### 4. Return the losses
        return results['loss_list']

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
        # 2. return output
        return np.dot(X, self.weights)

    """
    Add a bias/slope to the input
    """
    def add_bias(self, X):
        """
        Parameters:
        X:             Matrix of input with shape (number_of_examples, number_of_features). np.matrix
        """
        ### 1.Create a matrix with same row of X and 1 column more
        mat = np.zeros((X.shape[0], X.shape[1]+1))
        ### 2. Fill the matrix with the example with the bias feature (1)
        mat[:, :-1] = X
        mat[:, -1] = 1
        return mat

    """
    To implement if the threshold of the ooutput is neither 0.5 or 0
    """
    def get_threshold(self, X, y):
        raise NotImplementedError

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
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score

    ### Arguments
    parser = argparse.ArgumentParser(description='Variational Auto Encoder with MNIST dataset')

    parser.add_argument('--optim', type=str, default='gd',
                        help='type of optimization algorithm (gd, sgd, svrg)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='fixed stepsize')
                        
    args = parser.parse_args()



    print('MAIN\n-Solving a binary classification problem (-1, 1) with two classes of IRIS dataset')
    ### Loading and preparing IRIS data
    data = load_iris()
    inp, y = data['data'], data['target']
    X = inp[y<2]
    y = y[y<2]
    y[y==0] = -1
    y = np.reshape(y, (-1, 1))

    ### Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)

    ### Parameters
    optim_params = {'lr': 0.05}
    loss_params = {'weight_decay': 0.01}
    fit_params = {'epoch': 100, 'verbose': 0}
    print('-Displaying default parameters')
    print('\tLoss parameters:\t{}'.format(loss_params))
    print('\tOptim parameters:\t{}'.format(optim_params))
    print('\tModel parameters:\t{}'.format(fit_params))

    ### Init model, loss and optimzer
    model = Model(input_size = X.shape[1], init_weights='zeros')
    my_loss = loss.LogisticLoss(reg_coeff=loss_params['weight_decay'])
    optim = optimizer.GD(params=model.weights, loss= my_loss, learn_rate=optim_params['lr'])

    ### Shape info
    print('-Displaying shape of data and variable')
    print('\tInput: {}'.format(X.shape))
    print('\tTarget: {}'.format(y.shape))
    print('\tWeights (with bias): {}'.format(model.weights.shape))

    ### Fit the model
    loss_list = model.fit(X=X_train, y=y_train, optimizer=optim, epoch=100, verbose=0)

    ### Predict
    y_pred = model.predict(X_test)

    ### Display results
    negative = y_pred[y_test == -1]
    positive = y_pred[y_test == 1]
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    ax[0].plot(negative, 'o', label='negative')
    ax[0].plot(positive, 'o', label='positive')
    ax[0].legend()
    ax[0].set_title('Output distribution')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Output')
    ax[0].grid()

    ax[1].plot(loss_list)
    ax[1].set_title('Loss curve')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].grid()
    plt.show()
