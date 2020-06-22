import numpy as np
import functions as f

np.random.seed(10)


class NeuralNet:

    __slots__ = ["layer_shapes", "layers", "lr", "__fitted_once"]

    def __init__(self, layer_shapes):
        self.layer_shapes = [f.set_bias_as_weight(w) for w in layer_shapes]
        print(self.layer_shapes)
        self.layers = self.init_weights("he")
        self.lr = 0.002
        self.__fitted_once = False

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

    def _restart_weights(self):
        self.layers = self.init_weights("he")

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
        error = d = self.compute_output_layer_error(forwards[-1], y_true)
        wv = (1 / y_true.shape[0]) * np.dot(d.T, f.add_bias(forwards[-2]))
        wvs = [wv]

        # Compute weights variations based on error propagations
        for li in range(len(self.layers) - 1):
            d = np.dot(d, self.layers[-li - 1].T) * \
                f.add_bias(forwards[-li - 2]) * \
                (1 - f.add_bias(forwards[-li - 2]))
            d = d[:, :-1]
            wv = (1 / m) * np.dot(d.T, f.add_bias(forwards[-li - 3]))
            wvs.insert(0, wv)
        return wvs, error

    def train(self, x, y, epochs, batch_size, learning_rate, stochastic=True):
        if self.__fitted_once:
            print("Restarting weights to randomized initialization...")
            self._restart_weights()
        self.lr = learning_rate
        y = y.reshape(-1, 1)
        m = x.shape[0]
        n_batches = int(np.ceil(m / batch_size))

        # Iterate training by each epoch
        for epoch in range(epochs):
            batched_weights = []
            updated_weights = []
            accumulated_weights = [0] * len(self.layers)
            accumulated_error = []
            if stochastic:
                x, y = f.shuffle_vectors(x, y)

            for n in range(n_batches):
                # Get X and Y batches
                _x_batch = x[n * batch_size: (n + 1) * batch_size, :]
                _y_batch = y[n * batch_size: (n + 1) * batch_size]

                # Calculate backward propagation
                wvs, error = self.compute_backward_propagation(
                    _x_batch, _y_batch)
                accumulated_error.append(error.mean())
                batched_weights.append(wvs)

            for bw in range(len(batched_weights)):
                # bw -> List of lists with all weight matrices
                for wm in range(len(batched_weights[bw])):
                    accumulated_weights[wm] += batched_weights[bw][wm]

            mean_weight_vars = []
            for weights, wv in zip(self.layers, accumulated_weights):
                weights_ = weights - self.lr * wv.T
                mean_weight_vars.append(wv)
                updated_weights.append(weights_)

            self.__fitted_once = True
            self.layers = updated_weights

            error_avg = round(np.array(accumulated_error).mean(), 9)
            wvar = [round(np.array(mwv).mean(), 9) for mwv in mean_weight_vars]

            epoch_msg = f"""Epoch {epoch} | Error: {error_avg} | WV0: {wvar[0]} | WV1: {wvar[1]}"""
            print(epoch_msg)
