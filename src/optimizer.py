class Optimizer:

    def __init__(self, params, loss, learn_rate):
        self.params = params
        self.learn_rate = learn_rate
        self.lambda_reg = lambda_reg
        self.loss = loss

    def step(self, X, y):
        raise NotImplementedError

    def run(self, X, y):
        raise NotImplementedError


class GD(Optimizer):

    def step(self, X, y):
        raise NotImplementedError

    def run(self, X, y):
        raise NotImplementedError


class SGD(Optimizer):

    def step(self, X, y):
        raise NotImplementedError

    def run(self, X, y):
        raise NotImplementedError


class SVRG(Optimizer):

    def step(self, X, y):
        raise NotImplementedError

    def run(self, X, y):
        raise NotImplementedError
