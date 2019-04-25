from thex_data.data_clean import convert_class_vectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 300


class MCBaseModelVisualization:
    """
    Mixin Class for Multiclass BaseModel performance visualization
    """

    def save_roc_curve(self, i, roc_plots, model_classes):
        """ 
        Plot ROC curve for each class, but do not show. Plot to axis attached to class's plot (saved in roc_plots)
        """
        mean_fpr = np.linspace(0, 1, 100)
        class_probabilities = self.test_probabilities()
        y_test_vectors = convert_class_vectors(self.y_test, self.class_labels)
        for class_index, class_name in enumerate(self.class_labels):
            if class_name in model_classes:
                f, ax, tprs, aucs = roc_plots[class_name]
                column = class_probabilities[:, class_index]
                fpr, tpr, thresholds = roc_curve(
                    y_true=y_test_labels, y_score=column, sample_weight=None, drop_intermediate=True)
                # Updates FPR and TPR to nbe on the range 0 to 1 (for plotting)
                # Python directly alters list objects within dict.
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0  # Start curve at 0,0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                ax.plot(fpr, tpr, lw=1, alpha=0.3,
                        label='ROC fold %d (AUC=%0.2f)' % (i, roc_auc))

        return roc_plots

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
        # ax.plot(x, x, "--", label="Baseline")  # baseline
        plt.plot([0, 1], [0, 1], 'k--', label="Baseline")
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
