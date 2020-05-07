class Loss:

    def __init__(self, reg_coeff):
        self.reg_coeff = reg_coeff

    def compute_loss(self, X, y):
        raise NotImplementedError

    def compute_gradient(self, X, y):
        raise NotImplementedError


class LogisticLoss(Loss):

    def compute_loss(self, X, y):
        raise NotImplementedError

    def compute_gradient(self, X, y):
        raise NotImplementedError
