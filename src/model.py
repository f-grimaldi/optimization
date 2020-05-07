class Model:

    def __init__(self, input_size, init_weights=Model.init_random):
        self.input_size = input_size
        self.init_weigths = init_weigths


    def fit(self, X, y, optimizer, epoch, verbose = 0):
        self.weights = self.init_weigths(self.input_size)
        X = self.add_bias(X)
        results = optimizer.run(X, y, epoch, verbose)
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def add_bias(self, X):
        mat = np.zeros((X.shape[0], X.shape[1]+1))
        mat[:, :-1] = X
        mat[:, -1] = 1
        return mat

    def get_threshold(self, X, y):
        raise NotImplementedError

    @staticmethod
    def init_random(input_size):
        return np.random.randn(input_size+1, 1)

    @staticmethod
    def init_zeros(input_size):
        return np.zeros((input_size+1, 1))
