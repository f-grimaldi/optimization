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
        self.threshold = self.get_threshold(X, y)
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

    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score

    ### Arguments
    parser = argparse.ArgumentParser(description='Linear Classifier with Regularized Logistic Loss ')
    # 1. Optimizer params
    parser.add_argument('--optim', type=str, default='gd',
                        help='type of optimization algorithm (gd, sgd, svrg)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='fixed stepsize')
    # 2. Loss params
    parser.add_argument('--reg_coeff', type=float, default=0.001,
                        help='l2 regularization parameter')

    # 3. Model fit params
    parser.add_argument('--init', type=str, default='random',
                        help='type of iniztialization: "zeros" or "random"')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--verbose', type=int, default=0,
                        help='display info: 0=False, 1=True')

    # 4. Dataset Parameters
    parser.add_argument('--data', type=str, default='iris',
                        help='choose dataset (iris/mnist)')

    # 5. Seed
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)

    print('MAIN\n-Solving a binary classification problem (-1, 1) with two classes of {} dataset'.format(args.data.upper()))
    ### Loading and preparing data
    ### IRIS
    if args.data == 'iris':
        data = load_iris()
        inp, y = data['data'], data['target']
        X = inp[y<2]
        y = y[y<2]
        y[y==0] = -1
        y = np.reshape(y, (-1, 1))
    ### MNIST with digit 1 and 8
    elif args.data == 'mnist':
        from scipy.io import loadmat
        mnist = loadmat('MNIST.mat')
        X, y = mnist['input_images'], mnist['output_labels']
        X = X[(y.reshape(-1) == 1) | (y.reshape(-1) == 8)]
        y = y[(y == 1) | (y == 8)]
        y[y == 1] =  1
        y[y == 8] = -1
        y = np.reshape(y, (-1, 1))

    ### Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    ### Parameters
    optim_params = {'type': args.optim, 'lr': args.lr}
    loss_params = {'weight_decay': args.reg_coeff}
    fit_params = {'init_weights': args.init, 'epoch': args.epochs, 'verbose': args.verbose}
    print('-Displaying default parameters')
    print('\tLoss parameters:\t{}'.format(loss_params))
    print('\tOptim parameters:\t{}'.format(optim_params))
    print('\tModel parameters:\t{}'.format(fit_params))

    ### Init model, loss and optimzer
    model = Model(input_size = X.shape[1], init_weights=fit_params['init_weights'])
    my_loss = loss.LogisticLoss(reg_coeff=loss_params['weight_decay'])
    # Choose optimizer:
    if optim_params['type'] == 'gd':
        optim = optimizer.GD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'])
    elif optim_params['type'] == 'sgd':
        optim = optimizer.SGD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'])
    elif optim_params['type'] == 'svrg':
        optim = optimizer.SVRG(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'])
    else:
        raise NotImplementedError

    ### Shape info
    print('-Displaying shape of data and variable')
    print('\tInput: {}'.format(X.shape))
    print('\tTarget: {}'.format(y.shape))
    print('\tWeights (with bias): {}'.format(model.weights.shape))

    ### Fit the model and count time taken
    start =  time.time()
    results = model.fit(X=X_train, y=y_train, optimizer=optim, num_epochs=fit_params['epoch'], verbose=fit_params['verbose'])
    end = time.time()
    time_taken = end-start
    print('-Model fitted in {} second'.format(time_taken))

    ### Compute accuracy
    y_train_pred = model.predict(X_train)
    y_train_pred[y_train_pred >  0] =  1
    y_train_pred[y_train_pred <= 0] = -1
    y_pred = model.predict(X_test)
    y_pred[y_pred >  0] =  1
    y_pred[y_pred <= 0] = -1
    print('-Displaying Accuracies')
    print('\tTrain: {}'.format(accuracy_score(y_train, y_train_pred)))
    print('\tTest: {}'.format(accuracy_score(y_test, y_pred)))

    ### Display validation results
    ### Output distribution
    negative = y_pred[y_test == -1]
    positive = y_pred[y_test == 1]
    fig, ax = plt.subplots(3, 1, figsize=(20, 20))
    ax[0].plot(negative, 'o', label='negative')
    ax[0].plot(positive, 'o', label='positive')
    ax[0].legend()
    ax[0].set_title('Validation Output distribution')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Output')
    ax[0].grid()

    ### Loss curve
    ax[1].plot(results['loss_list'])
    ax[1].set_title('Loss curve')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].grid()

    ### Computing accuracy for each step
    X_testb = model.add_bias(X_test)
    accuracy_list = []

    for weights in results['params_list']:
        # 1. Get output
        y_pred = np.dot(X_testb, weights)
        # 2. Compute threshold and accuracy
        positive = y_pred[y_test ==  1]
        negative = y_pred[y_test == -1]
        # 2.a check which class is below threshold
        if np.min(negative) <= np.min(positive):
            threshold = np.mean([np.min(positive), np.max(negative)])
            y_pred[y_pred - threshold >  0] =  1
            y_pred[y_pred - threshold <= 0] = -1
        else:
            threshold = np.mean([np.min(negative), np.max(positive)])
            y_pred[y_pred - threshold >  0] = -1
            y_pred[y_pred - threshold <= 0] =  1
        # 2.b
        accuracy_list.append(accuracy_score(y_test, y_pred))

    ### Accuracy plot
    ax[2].plot(accuracy_list)
    ax[2].set_title('Validation Accuracy over step')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Accuracy')
    ax[2].grid()


    plt.show()
