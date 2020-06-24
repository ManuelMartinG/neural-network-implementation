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
