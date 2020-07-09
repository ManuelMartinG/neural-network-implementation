import numpy as np

try:
    from IPython import display
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("IPython and matplotlib are not managed dependencies."
                      + " If you want to use this visualization module,"
                      + " please, install them by yourself")


def plot_decision_function(x1, x2, neuralnet):
    nums = 300
    x1_ = np.linspace(x1.min(), x1.max(), nums)
    x2_ = np.linspace(x2.min(), x2.max(), nums)
    X1, X2 = np.meshgrid(x1_, x2_)
    flat_mesh = np.hstack([X1.flatten().reshape(-1, 1),
                          X2.flatten().reshape(-1, 1)])
    Z = neuralnet.forward_propagation(flat_mesh)[-1].reshape(nums, nums)
    plt.pcolormesh(X1, X2, Z, cmap=plt.cm.Spectral)


def plot_error_history(error_per_epoch, c="salmon",
                       linewidth=0.6, linestyle="-"):
    plt.plot(error_per_epoch, c=c, linewidth=linewidth, linestyle=linestyle)


class SubPlot2D:
    def __init__(self):
        self.nums = 300

    def make_mesh(self, x1, x2):
        x1_ = np.linspace(x1.min(), x1.max(), self.nums)
        x2_ = np.linspace(x2.min(), x2.max(), self.nums)
        X1, X2 = np.meshgrid(x1_, x2_)
        return X1, X2

    def flat_forward(self, X1, X2, forward):
        flat_mesh = np.hstack(
            [X1.flatten().reshape(-1, 1),
             X2.flatten().reshape(-1, 1)])
        Z = forward(flat_mesh)[-1].reshape(self.nums, self.nums)
        return Z

    def plot_gradient_space(self, ax_subplot, X1, X2, Z):
        ax_subplot.pcolormesh(X1, X2, Z, cmap=plt.cm.Spectral)

    def plot_error_decrease(self, ax_subplot, errors, epochs):
        ax_subplot.plot(epochs, errors, c="red")

    def make_subplots(self, x1, x2, acc_error, acc_epoch, forward):
        X1, X2 = self.make_mesh(x1, x2)
        display.clear_output(wait=True)
        fig, axs = plt.subplots(2)
        Z = self.flat_forward(X1, X2, forward)
        self.plot_gradient_space(axs[0], X1, X2, Z)
        self.plot_error_decrease(axs[1], acc_error, acc_epoch)

        fig.suptitle("Training Evolution")
        # plt.show();


class GifVisuals:
    def __init__(self, x, y):
        pass


class TrainingVisuals:

    MODES = ["SubPlot2D", "gif"]

    def __init__(self, mode, **kwargs):
        if mode not in self.MODES:
            raise ValueError(f"{mode} not supported")

        if mode == "SubPlot2D":
            self.visuals = SubPlot2D(**kwargs)
        elif mode == "gif":
            self.visuals = GifVisuals(**kwargs)

    def plot(self, func):
        return self.visuals.plot(func)

    @staticmethod
    def render():
        plt.show()
