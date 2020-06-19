import numpy as np
import tqdm


class Functions:
    """TODO: Put this in a separate module"""
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


f = Functions


class NeuralNet:

    __slots__ = ["layer_shapes", "hyper_params", "layers", "lr"]

    def __init__(self, layer_shapes):
        self.layer_shapes = [f.set_bias_as_weight(w) for w in layer_shapes]
        self.layers = self.init_weights("he")
        self.lr = 0.002

    def init_weights(self, method="he"):
        """TODO: Support Xavier, Uniform, Normal, etc."""
        layers = []
        for shape in self.layer_shapes:
            weights = f.he_initialize(shape)
            layers.append(weights)
        return layers

    @staticmethod
    def _forward_layer_pass(layer_weights, vector):
        """Forward Propagation over a specific Layer and Vector"""
        return f.sigmoid(np.dot(f.add_bias(vector), layer_weights))

    def compute_forward_propagation(self, x):
        """Compute Full Forward Propagation"""
        layer_outputs = [x]
        for layer in self.layers:
            x = self._forward_layer_pass(layer, x)
            layer_outputs.append(x)
        return layer_outputs

    @staticmethod
    def compute_output_layer_error(y_pred, y_true):
        d = (y_pred - y_true) * y_pred * (1 - y_pred)
        return d

    def compute_backward_propagation(self, x, y_true):
        """Compute full Backward Propagation"""
        m = x.shape[0]  # m -> Number of training samples
        forwards = self.compute_forward_propagation(x)

        # Compute error in output layer
        d = self.compute_output_layer_error(forwards[-1], y_true)
        wv = (1 / y_true.shape[0]) * np.dot(d.T, f.add_bias(forwards[-2]))
        wvs = [wv.T]

        # Compute weights variations based on error propagations
        for li in range(len(self.layers) - 1):
            d = np.dot(d, self.layers[-li - 1].T) * \
                f.add_bias(forwards[-li - 2]) * \
                (1 - f.add_bias(forwards[-li - 2]))
            d = d[:, :-1]
            wv = (1 / m) * np.dot(d.T, f.add_bias(forwards[-li - 3]))
            wvs.insert(0, wv)
        return wvs

    def train(self, x, y, epochs, batch_size):
        y = y.reshape(-1, 1)
        m = x.shape[0]
        n_batches = int(np.ceil(m / batch_size))

        # Iterate training by each epoch
        for epoch in range(epochs):
            batched_weights = []
            updated_weights = []
            accumulated_weights = [0] * len(self.layers)

            for n in tqdm.tqdm(range(n_batches)):
                # Get X and Y batches
                _x_batch = x[n * batch_size: (n + 1) * batch_size, :]
                _y_batch = y[n * batch_size: (n + 1) * batch_size]

                # Calculate backward propagation
                wvs = self.compute_backward_propagation(_x_batch, _y_batch)
                batched_weights.append(wvs)

            for bw in range(len(batched_weights)):
                # bw -> List of lists with all weight matrices
                for wm in range(len(batched_weights[bw])):
                    accumulated_weights[wm] += batched_weights[bw][wm]

            for weights, wv in zip(self.layers, accumulated_weights):
                weights = weights - self.lr * wv.T
                updated_weights.append(weights)
            self.layers = updated_weights
