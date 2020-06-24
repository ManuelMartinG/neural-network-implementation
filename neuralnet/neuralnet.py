import numpy as np
import time
import neuralnet.functions as f

try:
    import matplotlib.pyplot as plt
    from IPython import display
    pass
except Exception as ex:
    ImportError(ex)


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

    def __repr__(self):
        repr = "Neural Network Architecture\n" + \
               "===========================\n\n"
        for i, layer in enumerate(self.chain):
            layer_name = layer.__class__.__name__
            shapes = layer.shape
            repr += f"\n> Layer {i} - {layer_name} ({shapes[0]}, {shapes[1]})\n"
        return repr

    def __getitem__(self, index):
        return self.chain[index]

    def __len__(self):
        return len(self.chain)

    @property
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

    def add_weigth_diff(self, epoch, weight_diff):
        self.weight_variations_per_epoch[epoch].append(weight_diff)


class OptimizationVisuals:

    def __init__(self, x, y):
        self. n = 256
        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = y.min()
        self.ymax = y.max()
        self.x = np.linspace(self.xmin, self.xmax, self.n)
        self.y = np.linspace(self.ymin, self.ymax, self.n)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.positions = np.hstack(
            [self.X.flatten().reshape(-1, 1),
             self.Y.flatten().reshape(-1, 1)])

    def print_mesh(self, forward_prop_function):
        Z = forward_prop_function(self.positions)[-1]
        print(Z.mean())
        plt.gca().cla()
        plt.xlim((self.xmin, self.xmax))
        plt.ylim((self.ymin, self.ymax))
        plt.pcolormesh(self.X, self.Y, Z.reshape(256, 256), cmap=plt.cm.viridis)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.001)

    def decision_function_subplots(self):
        # TODO: Mostrar sólo 6 subplots dividiendo las epochs entre 6. Si hay menos epochs, se muestran
        # todas las que haya (pares)
        pass


class NeuralNet:

    def __init__(self, layer_shapes):
        self._no_bias_layer_shapes = layer_shapes
        self.layer_shapes = self._get_biased_absorbed_shapes(
                self._no_bias_layer_shapes)
        self.layers = self._build_layers()
        self.ophist = self._set_optim_history()
        self.training_mandatory_arguments = ["learning_rate", "batch_size"]
        self.__fitted__ = False

    def _set_optim_history(self):
        return OptimizationHistory(activate_mesh_history=True)

    def _build_layers(self):
        layers = []
        for shape in self.layer_shapes:
            fcl = FullyConnectedLayer(*shape, activation="sigmoid")
            fcl.initialize_weights(method="he")
            layers.append(fcl)
        return LayersChain(layers)

    def _get_biased_absorbed_shapes(self, layer_shapes):
        return [f.set_bias_as_weight(s) for s in layer_shapes]

    def _maybe_init_for_re_run(self):
        if self.__fitted__:
            self.layers = self._build_layers()
            self.ophist = self._set_optim_history()

    def forward_propagation(self, x):
        layer_outputs = [x]
        for layer in self.layers:
            x = layer.forward(x)
            layer_outputs.append(x)
        return layer_outputs

    def output_error(self, ypred, ytrue):
        delta = (ypred - ytrue) * ypred * (1 - ypred)
        return delta

    def backward_propagation(self, x, ytrue):
        m = x.shape[0]
        forwards = self.forward_propagation(x)

        # Get error on Output Layer (last layer)
        error = delta = self.output_error(forwards[-1], ytrue)
        output_weights_diff = (1/m) * np.dot(delta.T, f.add_bias(forwards[-2]))

        all_weights_diff = [output_weights_diff]

        # Get error on Hidden Layers (all layers except last one)
        for layer_pos in range(len(self.layers) - 1):
            delta = np.dot(delta, self.layers[-layer_pos - 1].weights.T) *\
                    f.add_bias(forwards[-layer_pos - 2]) * (1 - f.add_bias(forwards[-layer_pos - 2]))

            delta = delta[:, :-1]  # Avoid propagate bias

            layer_weights_diff = (1/m) * np.dot(delta.T, f.add_bias(forwards[-layer_pos - 3]))
            all_weights_diff.insert(0, layer_weights_diff)
        return all_weights_diff, error

    @staticmethod
    def show_progress(epoch, error_avg, weight_diffs, show_after=5):
        if epoch % show_after == 0:
            print(f"Epoch {epoch} | "
                  + f"Error: {error_avg} | "
                  + f"WV0: {weight_diffs[0]} | "
                  + f"WV1: {weight_diffs[1]}")

    def validate_optimization_arguments(self, **kwargs):
        for arg in self.training_mandatory_arguments:
            if arg not in kwargs.keys():
                raise ValueError(f"Argument {arg} is mandatory!")

    def optimize(self, x, y, epochs, **kwargs):
        # Validate input parameters and restart state if needed
        self.validate_optimization_arguments(**kwargs)
        self._maybe_init_for_re_run()
        ovisuals = OptimizationVisuals(x=x[:, 0], y=x[:, 1])

        # Set training parameters
        y = y.reshape(-1, 1)
        m = x.shape[0]
        learning_rate = kwargs.get("learning_rate", 0.002)
        batch_size = kwargs["batch_size"]
        n_batches = int(np.ceil(m / batch_size))
        progess_cadence = kwargs.get("show_after", 10)
        self.ophist.weight_variations_per_epoch = [0] * epochs

        for epoch in range(epochs):
            batched_weights = []
            updated_weights = []
            accumulated_weights = [0] * len(self.layers)

            # Shuffle samples for every epoch in order to perform stochastic optimization
            if kwargs.get("stochastic", True):
                x, y = f.shuffle_vectors(x, y)

            for n in range(n_batches):
                x_batch = x[n * batch_size: (n + 1) * batch_size, :]
                y_batch = y[n * batch_size: (n + 1) * batch_size]

                weights_diff, error = self.backward_propagation(
                        x_batch, y_batch)

                # Save Mean Error and Weight Variations
                self.ophist.add_error(error.mean())
                batched_weights.append(weights_diff)

            # Sum up for each weight matrices, their respective diffs for each batch
            for bw in range(len(batched_weights)):
                for wm in range(len(batched_weights[bw])):
                    accumulated_weights[wm] += batched_weights[bw][wm]

            self.ophist.weight_variations_per_epoch[epoch] = []
            for weights, weights_diff in zip(self.layers.weights, accumulated_weights):
                new_weights = weights - learning_rate * weights_diff.T

                # Save Weight differences applied
                self.ophist.add_weigth_diff(epoch, weights_diff)

                # Update new Neural Network Weights
                updated_weights.append(new_weights)

            self.layers.update_weights(updated_weights)
            self.__fitted__ = True

            if kwargs.get("visuals", True):
                ovisuals.print_mesh(self.forward_propagation)

            error_avg = round(np.array(self.ophist.mean_error_per_epoch).mean(), 9)
            wvar = [round(np.array(mwv).mean(), 9) for mwv in self.ophist.weight_variations_per_epoch[epoch]]

            self.show_progress(epoch=epoch, error_avg=error_avg,
                               weight_diffs=wvar, show_after=progess_cadence)
        if kwargs.get("visuals", True):
            plt.show()
