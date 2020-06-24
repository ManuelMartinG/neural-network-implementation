import numpy as np

import neuralnet.functions as f
from neuralnet.layers import FullyConnectedLayer, LayersChain
from neuralnet.plotter import TrainingVisuals


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


class NeuralNet:

    def __init__(self, layer_shapes, **kwargs):
        self._no_bias_layer_shapes = layer_shapes
        self.layer_shapes = self._get_biased_absorbed_shapes(
                self._no_bias_layer_shapes)
        self.layers = self._build_layers()
        self.ophist = self._set_optim_history()
        self.training_mandatory_arguments = ["learning_rate", "batch_size"]
        self.__fitted__ = False

        self._handle_kwargs(**kwargs)

    def _handle_kwargs(self, **kwargs):
        self.visual_mode = kwargs.get("visual_mode", False)

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

        y = y.reshape(-1, 1)
        m = x.shape[0]
        learning_rate = kwargs.get("learning_rate", 0.002)
        batch_size = kwargs["batch_size"]
        n_batches = int(np.ceil(m / batch_size))
        progess_cadence = kwargs.get("show_after", 10)
        self.ophist.weight_variations_per_epoch = [0] * epochs

        if self.visual_mode:
            visuals = TrainingVisuals(self.visual_mode, x=x[:, 0], y=x[:, 1])

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

            if self.visual_mode:
                visuals.plot(self.forward_propagation)

            error_avg = round(np.array(self.ophist.mean_error_per_epoch).mean(), 9)
            wvar = [round(np.array(mwv).mean(), 9) for mwv in self.ophist.weight_variations_per_epoch[epoch]]

            self.show_progress(epoch=epoch, error_avg=error_avg,
                               weight_diffs=wvar, show_after=progess_cadence)
        if self.visual_mode:
            pass
            #visuals.render()
