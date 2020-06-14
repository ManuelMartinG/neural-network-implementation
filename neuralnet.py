import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.datasets import make_blobs


class Maths:
    @staticmethod
    def he_initialize(shape):
        he = np.random.normal(loc=0, scale=np.sqrt(1 / shape[0]))
        return he * np.random.randn(*shape)

    @staticmethod
    def set_bias_as_weight(shape):
        return shape[0] + 1, shape[1]

    @staticmethod
    def add_bias(vector):
        return np.hstack([vector, np.ones((vector.shape[0], 1))])

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class NeuralNet:
    
    def __init__(self, layers, **hyperparams):
        self.m = Maths()
        self.layers_sizes = [self.m.set_bias_as_weight(sh) for sh in layers]
        self.layers = self.initialize_layer_weights()
        self.lr = hyperparams.get("lr", 0.01)
        
    def initialize_layer_weights(self):
        layers = []
        for size in self.layers_sizes:
            layers.append(self.m.he_initialize(size))
        return layers

    def _forward_layer(self, layer, v):
        return self.m.sigmoid(np.dot(self.m.add_bias(v), layer))

    def get_layer_forwards(self, x):
        layer_forwards = [x]
        for layer in self.layers:
            x = self._forward_layer(layer, x)
            layer_forwards.append(x)
        return layer_forwards
    
    def backward(self, x, ytrue):
        m = x.shape[0]
        forwards = self.get_layer_forwards(x)
        
        # Output Layer Error
        d = (forwards[-1] - ytrue) * forwards[-1] * (1 - forwards[-1])
        wv = (1 / m) * np.dot(d.T, self.m.add_bias(forwards[-2]))
        
        ds = [d]
        wvs = [wv.T]
        
        for l in range(len(self.layers) - 1):
            d = np.dot(d, self.layers[-l - 1].T) * \
                   self.m.add_bias(forwards[-l - 2]) * \
                   (1 - self.m.add_bias(forwards[-l - 2]))
            d = d[:, :-1]
            wv = (1/m) * np.dot(d.T, self.m.add_bias(forwards[-l - 3]))
            ds.insert(0, d)
            wvs.insert(0, wv.T)
        return wvs
    
    def infer(self, x):
        for layer in self.layers:
            x = self._forward_layer(layer, x)
        return x
    
    def train(self, x, y, epochs, batch_size):
        y = y.reshape(-1, 1)
        m = x.shape[0]
        n_batches = int(np.floor(m / batch_size))
        rem_batch_size = m - n_batches * batch_size
        
        for _ in tqdm.tqdm(range(epochs)):
            batched_weights = []
            new_weights = []
            accumulated_weights = [0] * len(self.layers)
            
            for n in range(n_batches):
                
                x_ = x[n * batch_size: (n + 1)* batch_size]
                y_ = y[n * batch_size: (n + 1) * batch_size]

                wvs = self.backward(x_, y_)
                batched_weights.append(wvs)
            
            # Sum all weights variations for each batch
            for i in range(0, len(batched_weights)):
                for j in range(len(batched_weights[i])):
                    accumulated_weights[j] = accumulated_weights[j] + batched_weights[i][j]
            
            # For the weights in each accumulated batched weights variations, apply update
            for weigths, wv in zip(self.layers, accumulated_weights):
                weigths = weigths - self.lr * wv
                new_weights.append(weigths)
            self.layers = new_weights
        return self.layers
    