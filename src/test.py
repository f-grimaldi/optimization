if __name__ == '__main__':
  import model
  import optimizer
  import loss

  import argparse
  import time

  import numpy as np
  import matplotlib.pyplot as plt
  from tqdm import tqdm
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
  parser.add_argument('--iter_epoch', type=int, default=10,
                      help='iter epoch used only in svrg')
  # 2. Loss params
  parser.add_argument('--reg_coeff', type=float, default=0.001,
                      help='l2 regularization parameter')

  # 3. Model fit params
  parser.add_argument('--init', type=str, default='zeros',
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
      X = inp[y < 2]
      y = y[y < 2]
      y[y == 0] = -1
      y = np.reshape(y, (-1, 1))
  ### MNIST with digit 1 and 7
  elif args.data == 'mnist':
      from scipy.io import loadmat
      mnist = loadmat('MNIST.mat')
      X, y = mnist['input_images'], mnist['output_labels']
      X = X[(y.reshape(-1) == 1) | (y.reshape(-1) == 7)]
      y = y[(y == 1) | (y == 7)]
      y[y == 1] = 1
      y[y == 7] = -1
      y = np.reshape(y, (-1, 1))

  ### Splitting dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

  ### Parameters
  optim_params = {'type': args.optim, 'lr': args.lr, 'iter_epoch': args.iter_epoch}
  loss_params = {'weight_decay': args.reg_coeff}
  fit_params = {'init_weights': args.init, 'epoch': args.epochs, 'verbose': args.verbose}
  print('-Displaying default parameters')
  print('\tLoss parameters:\t{}'.format(loss_params))
  print('\tOptim parameters:\t{}'.format(optim_params))
  print('\tModel parameters:\t{}'.format(fit_params))

  ### Init model, loss and optimzer
  model = model.Model(input_size=X.shape[1], init_weights=fit_params['init_weights'])
  my_loss = loss.LogisticLoss(reg_coeff=loss_params['weight_decay'])
  # Choose optimizer:
  if optim_params['type'] == 'gd':
      optim = optimizer.GD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'])
  elif optim_params['type'] == 'sgd':
      optim = optimizer.SGD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'])
  elif optim_params['type'] == 'sag':
      optim = optimizer.SGD(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'])
  elif optim_params['type'] == 'svrg':
      optim = optimizer.SVRG(params=model.weights, loss=my_loss, learn_rate=optim_params['lr'], iter_epoch=optim_params['iter_epoch'])
  else:
      raise NotImplementedError

  ### Shape info
  print('-Displaying shape of data and variable')
  print('\tInput: {}'.format(X.shape))
  print('\tTarget: {}'.format(y.shape))
  print('\tWeights (with bias): {}'.format(model.weights.shape))

  ### Fit the model and count time taken
  start = time.time()
  results = model.fit(X=X_train, y=y_train, optimizer=optim, num_epochs=fit_params['epoch'], verbose=fit_params['verbose'])
  end = time.time()
  time_taken = end-start
  print('-Model fitted in {} second'.format(time_taken))

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
  print('\tTest: {}'.format(accuracy_score(y_test, y_pred)))


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
  print('If it takes too long maybe is beacuse using SGD with a discrete number of epochs')
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

  if args.data == 'mnist':
      fig, ax = plt.subplots(4, 4)
      for i in range(4):
          for j in range(4):
              n = int(np.random.randint(0, X_test.shape[0], (1, 1)))
              x, y = X_testb[n, :], y_test[n]
              ax[i, j].imshow(x[:-1].reshape(28, 28), cmap='gray')
              output = float(np.dot(x, results['params_list'][-1]))
              if output > 0:
                  ax[i, j].set_title('Predicted 1 with probability {}'.format(1/(1 + np.exp(-output))))
              else:
                  ax[i, j].set_title('Predicted 7 with probability {}'.format(1 - 1/(1 + np.exp(-output))))

  plt.show()
