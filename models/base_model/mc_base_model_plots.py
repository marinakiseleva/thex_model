from thex_data.data_clean import convert_class_vectors
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 300


class MCBaseModelVisualization:
    """
    Mixin Class for Multiclass BaseModel performance visualization
    """

    def plot_mc_roc_curves(self, models):
        """ 
        Plot using Sklearn ROC Curve logic
        :param models: Dictionary of class name to model. if models[class_name] is None, that ROC curve will not be plotted.
        """
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        # class_probabilities is a numpy ndarray with a row for each X_test
        # sample, and a column for each class probability, in order of valid
        # self.class_labels
        class_probabilities = self.test_probabilities()
        # y_test_vectors has TARGET_LABEL column, with each class vector of length
        # self.class_labels
        y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
        for class_index, class_name in enumerate(self.class_labels):
            # If there is a valid model for this class
            if models[class_name] is not None:
                column = class_probabilities[:, class_index]
                y_test_labels = self.relabel(class_index, y_test_vectors)

                fpr, tpr, thresholds = roc_curve(
                    y_true=y_test_labels, y_score=column, sample_weight=None, drop_intermediate=True)

                plt.plot(fpr, tpr, label=class_name)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.plot(x, x, "--", label="Baseline")  # baseline
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
