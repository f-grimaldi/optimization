if __name__ == '__main__':
    import model
    import optimizer
    import loss

    import argparse
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from sklearn.utils import shuffle
    from tabulate import tabulate

    ### Arguments
    parser = argparse.ArgumentParser(description='Linear Classifier with Regularized Logistic Loss ')
    # 1. Optimizer params
    parser.add_argument('--optim', type=str, default='gd',
                        help='type of optimization algorithm (gd, sgd, sag, svrg)')
    parser.add_argument('--lr', type=float, default=0.7,
                        help='fixed stepsize')
    parser.add_argument('--tollerance', type=float, default=0.001,
                        help='stopping criterion tollerance')
    parser.add_argument('--iter_epoch', type=int, default=10,
                        help='iter epoch used only in svrg')

    # 2. Loss params
    parser.add_argument('--reg_coeff', type=float, default=0.001,
                        help='l2 regularization parameter')

    # 3. Model fit params
    parser.add_argument('--init', type=str, default='zeros',
                        help='type of initialization: "zeros" or "random"')
    parser.add_argument('--epochs', type=int, default=1000,
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

    print('MAIN\n-Solving a binary classification problem (-1, 1) for a9a dataset')
    ### Loading and preparing data
    X_train = np.genfromtxt('../a9a/a9a_train.csv', delimiter=',', skip_header=True)[:, 1:]
    y_train = np.genfromtxt('../a9a/a9a_train.csv', delimiter=',', skip_header=True)[:, 0].reshape(-1, 1)
    X_test = np.genfromtxt('../a9a/a9a_test.csv', delimiter=',', skip_header=True)[:, 1:]
    y_test = np.genfromtxt('../a9a/a9a_test.csv', delimiter=',', skip_header=True)[:, 0].reshape(-1, 1)


    # Balance
    length = np.sum(y_train == 1)
    idx1 = y_train.reshape(-1) == 1
    idx0 = y_train.reshape(-1) == -1
    X_train = np.concatenate((X_train[idx1, :], X_train[idx0, :][:length]), axis=0)
    y_train = np.concatenate((y_train[idx1, :], y_train[idx0, :][:length]), axis=0)

    # Shuffle
    X_train, y_train = shuffle(X_train, y_train)


    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)



    ### Init model, loss and optimzer

    ### Parameters
    optim_params = {'type': args.optim, 'lr': args.lr, 'iter_epoch': args.iter_epoch, 'tollerance': args.tollerance}
    loss_params = {'weight_decay': args.reg_coeff}
    fit_params = {'init_weights': args.init, 'epoch': args.epochs, 'verbose': args.verbose}
    print('-Displaying default parameters')
    print('\tLoss parameters:\t{}'.format(loss_params))
    print('\tOptim parameters:\t{}'.format(optim_params))
    print('\tModel parameters:\t{}'.format(fit_params))

    model = model.Model(input_size=X.shape[1], init_weights=fit_params['init_weights'])
    my_loss = loss.LogisticLoss(reg_coeff=loss_params['weight_decay'])
    # Choose optimizer:
    if optim_params['type'] == 'gd':
      optim = optimizer.GD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'], tollerance=optim_params['tollerance'])
    elif optim_params['type'] == 'sgd':
      optim = optimizer.SGD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'], tollerance=optim_params['tollerance'])
    elif optim_params['type'] == 'sag':
      optim = optimizer.SGD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'], tollerance=optim_params['tollerance'])
    elif optim_params['type'] == 'svrg':
      optim = optimizer.SVRG(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'], tollerance=optim_params['tollerance'], iter_epoch=optim_params['iter_epoch'])
    else:
      raise NotImplementedError

    ### Shape info
    print('-Displaying shape of data and variable')
    print('\tInput: {}'.format(X.shape))
    print('\tTarget: {}'.format(y.shape))
    print('\tWeights (with bias): {}\n'.format(model.weights.shape))

    ### Fit the model and count time taken
    start = time.time()
    results = model.fit(X=X_train, y=y_train, optimizer=optim, num_epochs=fit_params['epoch'], verbose=fit_params['verbose'])
    end = time.time()
    time_taken = end-start
    print('-Model fitted in {} second\n'.format(time_taken))

    ### Output probability distribution
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    negative = 1/(1+np.exp(-y_pred[y_test == -1]))
    positive = 1/(1+np.exp(-y_pred[y_test == 1]))

    ### Compute accuracy
    y_train_pred[y_train_pred > 0] = 1
    y_train_pred[y_train_pred <= 0] = -1
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    print('-Displaying Accuracies')
    print('\tTrain: {}'.format(accuracy_score(y_train, y_train_pred)))
    print('\tTest: {}\n'.format(accuracy_score(y_test, y_pred)))

    ### Display validation results
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
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Loss')
    ax[1].grid()

    ### Computing accuracy for each step
    X_testb = model.add_bias(X_test)
    accuracy_list = []

    print('Starting to compute accuracy on every step.')
    print('If it takes too long maybe is because using SGD with a discrete number of epochs')
    for weights in tqdm(results['params_list']):
        # 1. Get output
        y_pred = np.dot(X_testb, weights)
        # 2. Compute accuracy
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = -1
        accuracy_list.append(accuracy_score(y_test, y_pred))

    ### Accuracy plot
    ax[2].plot(accuracy_list)
    ax[2].set_title('Validation Accuracy over step')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Accuracy')
    ax[2].grid()

    # Display Loss and Time at each Epoch
    disp = []
    for i in range(len(results["loss_list"])):
      if (i+1) % 10 == 0:
        disp.append([i+1, results["loss_list"][i], results["time_list"][i]])
    print(tabulate(disp, headers=['Epoch', 'Loss', 'Time']))


    plt.scatter(results["time_list"], results["loss_list"], c='r', s=15)
    plt.plot([float(i) for i in results["time_list"]], results["loss_list"], c='r')
    plt.xlabel('Time')
    plt.ylabel('Loss function')
    plt.title('{} with reg_coef={}'.format(optim_params['type'], loss_params['weight_decay']))
    plt.grid()
    plt.show()
