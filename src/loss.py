import numpy as np

class Loss:

    def __init__(self, reg_coeff):
        self.reg_coeff = reg_coeff

    def compute_loss(self, X, y, W):
        """
        X: matrix of input organized in the following way: number_of_input, number_of_features
        y: array of target organized in the following way: number_of_input, 1
        W: array of parameters organized in the following way: number_of_features, 1
        """
        raise NotImplementedError

    def compute_gradient(self, X, y, W):
        """
        X: matrix of input organized in the following way: number_of_input, number_of_features
        y: array of target organized in the following way: number_of_input, 1
        W: array of parameters organized in the following way: number_of_features, 1
        """
        raise NotImplementedError


class LogisticLoss(Loss):

    def compute_loss(self, X, y, W):

        """
        X: matrix of input organized in the following way: number_of_input, number_of_features
        y: array of target organized in the following way: number_of_input, 1
        W: array of parameters organized in the following way: number_of_features, 1
        """

        activation = np.dot(X, W)
        regularization = 0.5 * self.reg_coeff * np.linalg.norm(W) ** 2
        log_argument = 1 + np.exp(-y.transpose() * activation.reshape(-1))
        loss = np.mean(np.log(log_argument)) + regularization

        return loss

    def compute_gradient(self, X, y, W):

        """
        X: matrix of input organized in the following way: number_of_input, number_of_features
        y: array of target organized in the following way: number_of_input, 1
        W: array of parameters organized in the following way: number_of_features, 1
        """

        activation = np.dot(X, W)
        sigmoid = 1 / (1 + np.exp(-y.transpose() * activation.reshape(-1)))
        regularization = self.reg_coeff * W
        target_x_output = y.transpose() * (1-sigmoid)

        if X.shape[0] > 1:
            gradient = -np.mean(target_x_output.transpose()*X, axis=0).reshape(-1, 1) + regularization
        else:
            gradient = -target_x_output.transpose() * X.transpose() + regularization


        # print('y*(1-sigmoid(y*XW)) is: {}'.format(target_x_output))
        # print('Target*output*X is {}'.format(target_x_output.transpose()*X))
        # print('Gradient.T is: {}'.format(gradient.transpose()))
        # print('Gradient shape is: {}'.format(gradient.shape))
        # print('---------------------------------')

        return gradient


if __name__ == "__main__":
    X = np.random.randint(0, 10, (10, 3))
    y = np.random.randint(0, 2, (10, 1))
    W = np.random.randint(0, 5, (3, 1))

    print('Initial shape:')
    print('\tX: {}'.format(X.shape))
    print('\ty: {}'.format(y.shape))
    print('\tW: {}\n'.format(W.shape))

    Loss = LogisticLoss(1)
    loss = Loss.compute_loss(X, y, W)
    gradient = Loss.compute_gradient(X, y, W)
    print('Results with regularization coefficient set to 1:')
    print('\tLoss: {}\t\t'.format(loss))
    print('\tGradient (transpose): {}\t\t with shape {}'.format(gradient.transpose(), gradient.shape))
