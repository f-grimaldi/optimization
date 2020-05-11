if __name__ == '__main__':
  import model
  import optimizer
  import loss

  import argparse
  import time

  import numpy as np
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
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

  ### Parameters
  optim_params = {'type': args.optim, 'lr': args.lr}
  loss_params = {'weight_decay': args.reg_coeff}
  fit_params = {'init_weights': args.init, 'epoch': args.epochs, 'verbose': args.verbose}
  print('-Displaying default parameters')
  print('\tLoss parameters:\t{}'.format(loss_params))
  print('\tOptim parameters:\t{}'.format(optim_params))
  print('\tModel parameters:\t{}'.format(fit_params))

  ### Init model, loss and optimzer
  model = model.Model(input_size = X.shape[1], init_weights=fit_params['init_weights'])
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
  print('-Weights are: {}'.format(model.weights))

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
