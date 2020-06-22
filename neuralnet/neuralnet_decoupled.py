import numpy as np
import neuralnet.functions as f


class FullyConnectedLayer:
    def __init__(self, shape_in, shape_out, activation):
        self.shape = (shape_in, shape_out)

        if activation == "sigmoid":
            self.activation = f.sigmoid
        else:
            raise ValueError(f"""{activation} not currently supported.
                             Try with `sigmoid`""")

        self.weights = None  # To be initialized by NeuralNet

    def initialize_weights(self, method="he"):
        if method == "he":
            self.weights = f.he_initialize(self.shape)
        else:
            raise ValueError("Only supported He-Normal method")

    def forward(self, x):
        z = np.dot(f.add_bias(x), self.weights)
        return self.activation(z)

    def backpropagate(self):
        pass


class LayersChain():
    def __init__(self, layers_list):
        self.chain = layers_list

    def __getitem__(self, index):
        return self.chain[index]

    def __len__(self):
        return len(self.chain)

    def weights(self):
        return [l.weights for l in self.chain]

    def update_weights(self, new_weights):
        new_chain = []
        for layer, new_weights in zip(self.chain, new_weights):
            layer.weights = new_weights
            new_chain.append(layer)
        self.chain = new_chain


class OptimizationHistory:
    def __init__(self, activate_mesh_history=False):
        self.mean_error_per_epoch = []
        self.weight_variations_per_epoch = []

        if activate_mesh_history:
            self.mesh_predictions_per_epoch = []

    def add_error(self, error):
        self.mean_error_per_epoch.append(error)

    def add_weigth_diff(self, weight_diff):
        self.weight_variations_per_epoch.append(weight_diff)


class NeuralNet:
    def __init__(self, layer_shapes):
        self.layer_shapes = [f.set_bias_as_weight(s) for s in layer_shapes]
        self.layers = self._build_layers()
        self.ophist = self._set_optim_history()
        self.training_mandatory_arguments = [
            "learning_rate",
            "batch_size"
        ]

    def _set_optim_history(self):
        return OptimizationHistory(activate_mesh_history=True)

    def _build_layers(self):
        layers = []
        for shape in self.layer_shapes:
            fcl = FullyConnectedLayer(*shape, activation="sigmoid")
            fcl.initialize_weights(method="he")
            layers.append(fcl)
        return LayersChain(layers)

    def forward_propagation(self, x):
        layer_outputs = []
        for layer in self.layers:
            x = layer.forward(x)
            layer_outputs.append(x)
        return layer_outputs

    def output_error(ypred, ytrue):
        delta = (ypred - ytrue) * ypred * (1 - ypred)
        return delta

    def backward_propagation(self, x, ytrue):
        m = x.shape[0]
        forwards = self.forward_propagation(x)

        error = delta = self.output_error(forwards[-1], ytrue)
        output_weights_diff = (1/m) * np.dot(delta.T, f.add_bias(forwards[-2]))
        all_weights_diff = [output_weights_diff]

        for layer_pos in range(len(self.layers) - 1):
            delta = np.dot(
                delta, self.layers[-layer_pos - 1].weights.T
                    ) * f.add_bias(forwards[-layer_pos - 2]) * \
                        (1 - f.add_bias(forwards[-layer_pos - 2]))
            delta = delta[:, :-1]  # Avoid propagate bias
            layer_weights_diff = (1/m) * np.dot(
                delta.T, f.add_bias(forwards[-layer_pos - 3])
            )
            all_weights_diff.append(layer_weights_diff)
        return all_weights_diff, error

    @staticmethod
    def show_progress(epoch, error, weight_diffs, show_after=5):
        epoch_msg = f"""Epoch {epoch} | Error: {error} | WV0: {weight_diffs[0]} | WV1: {weight_diffs[1]}"""
        if epoch % show_after == 0:
            print(epoch_msg)

    def validate_optimization_arguments(self, **kwargs):
        for arg in self.training_mandatory_arguments:
            if arg not in kwargs.keys():
                raise ValueError(f"Argument {arg} is mandatory!")

    def optimize(self, x, y, epochs, **kwargs):
        self.validate_optimization_arguments(**kwargs)
        y = y.reshape(-1, 1)
        m = x.shape[0]
        batch_size = kwargs["batch_size"]
        n_batches = int(np.ceil(m / batch_size))

        for epoch in range(epochs):
            batched_weights = []
            updated_weights = []
            accumulated_weights = [0] * len(self.layers)

            # Gradient Descent
            if kwargs.get("stochastic", True):
                x, y = f.shuffle_vectors(x, y)

            for n in range(n_batches):
                x_batch = x[n * batch_size: (n + 1) * batch_size, :]
                y_batch = y[n * batch_size: (n + 1) * batch_size]

                weights_diff, error = self.backward_propagation(
                    x_batch, y_batch)

                self.ophist.add_error(error.mean())
                batched_weights.append(weights_diff)

            # Sum up for each weight matrices, their respective diffs for each batch
            for bw in range(len(batched_weights)):
                for wm in range(len(batched_weights[bw])):
                    accumulated_weights[wm] += batched_weights[bw][wm]

            for weights, weights_diff in zip(self.layers.weights, accumulated_weights):
                new_weights = weights - \
                    kwargs["learning_rate"] * weights_diff.T
                self.ophist.add_weigth_diff(new_weights)
                updated_weights.append(new_weights)

            self.layers.update_weights(updated_weights)

            error_avg = round(
                np.array(self.ophist.mean_error_per_epoch).mean(), 9)
            wvar = [round(np.array(mwv).mean(), 9) for mwv in self.ophist.weight_variations_per_epoch]

            self.show_progress(
                epoch=epoch, error_avg=error_avg, weight_diffs=wvar,
                show_after=kwargs.get("show_after", 10)
                )
