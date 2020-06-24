import numpy as np
import time

try:
    from IPython import display
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("IPython and matplotlib are not managed dependencies."
                      + " If you want to use this visualization module,"
                      + " please, install them by yourself")


class JupyterInline:
    def __init__(self, x, y):
        self. n = 256
        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = y.min()
        self.ymax = y.max()
        print(self.xmin)
        print(self.xmax)
        # Create x and y space
        self.x_space = np.linspace(self.xmin, self.xmax, self.n)
        self.y_space = np.linspace(self.ymin, self.ymax, self.n)

        self.X_mesh, self.Y_mesh = np.meshgrid(self.x_space, self.y_space)
        self.positions = np.hstack(
            [self.X_mesh.flatten().reshape(-1, 1),
             self.Y_mesh.flatten().reshape(-1, 1)])

    def plot(self, forward_prop_function):
        Z = forward_prop_function(self.positions)[-1]
        plt.gca().cla()
        plt.xlim((self.xmin, self.xmax))
        plt.ylim((self.ymin, self.ymax))
        plt.pcolormesh(self.X_mesh, self.Y_mesh, Z.reshape(
            256, 256), cmap=plt.cm.Spectral)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.00001)


class GifVisuals:
    def __init__(self, x, y):
        pass


class TrainingVisuals:

    MODES = ["inline", "gif"]

    def __init__(self, mode, **kwargs):
        if mode not in self.MODES:
            raise ValueError(f"{mode} not supported")

        if mode == "inline":
            self.visuals = JupyterInline(**kwargs)
        elif mode == "gif":
            self.visuals = GifVisuals(**kwargs)

    def plot(self, func):
        return self.visuals.plot(func)

    @staticmethod
    def render():
        plt.show()
