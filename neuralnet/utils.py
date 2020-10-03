
def show_training_progression():
    """Generates a plot with train and validation loss plot.
    """
    pass


def pretty_print_nn():
    """Takes a NeuralNet class and displays it as an HTML div.
    """
    pass


class EarlyStopper:

    def __init__(self, metric, patience, tolerance):
        self.metric_to_watch = metric
        self.patience = patience
        self.tolerance = tolerance

    def _monitor(self):
        """Start watching the defined metric.
        """
        pass

    def _evaluate_policy(self):
        """Checks wether if patience and tolerance have been met.
        """
        pass

    def _stop_training(self):
        pass
