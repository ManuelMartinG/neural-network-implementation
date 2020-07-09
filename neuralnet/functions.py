import numpy as np


def he_initialize(shape):
    he = np.random.normal(loc=0, scale=np.sqrt(1 / shape[0]))
    return he * np.random.randn(*shape)


def set_bias_as_weight(shape):
    return shape[0] + 1, shape[1]


def add_bias(vector):
    return np.hstack([vector, np.ones((vector.shape[0], 1))])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def shuffle_vectors(x, y):
    rd = np.arange(len(x))
    np.random.shuffle(rd)
    x = x[rd]
    y = y[rd]
    return x, y


def _stable_clip(x):
    """Used to avoid numerical inestability when"""
    return np.clip(x, 1e-7, 1 - 1e-7)


def mean_squared_error(ypred, ytrue):
    return (ypred - ytrue) * ypred * (1 - ypred)


def cross_entropy(ypred, ytrue, binary=True):
    # return -ytrue * np.log(_stable_clip(ypred)) -\
    #         (1 - ytrue) * np.log(1 - _stable_clip(ypred))
    return _stable_clip(ypred) - ytrue


def kullback_leibler_divergence():
    pass


def hinge_loss():
    pass
