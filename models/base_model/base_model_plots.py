import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

from thex_data.data_consts import code_cat


class BaseModelVisualization:
    """
    Mixin Class for BaseModel performance visualization
    """

    def plot_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = confusion_matrix(
            self.y_test, self.predictions, labels=np.unique(self.y_test))

        classes = [code_cat[cc] for cc in np.unique(self.y_test)]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
        else:
            title = "Confusion Matrix (without normalization)"

        rcParams['figure.figsize'] = 8, 8
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def plot_class_accuracy(self, class_accuracies, plot_title):
        """
        Plots class accuracies as percentages only. Used for CV runs.
        :param class_accuracies: Mapping of classes to accuracies
        """
        transient_classes, accuracies = [], []
        for c in class_accuracies.keys():
            transient_classes.append(code_cat[c])
            accuracies.append(class_accuracies[c])
        rcParams['figure.figsize'] = 8, 8
        class_index = np.arange(len(transient_classes))
        plt.bar(class_index, accuracies)
        plt.xticks(class_index, transient_classes, fontsize=12, rotation=30)
        plt.yticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
        plt.title(plot_title, fontsize=15)
        plt.xlabel('Transient Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.show()

    def compute_plot_class_accuracy(self, class_accuracies, class_counts):
        """
        Computes and Visualizes accuracy per class with bar graph
        """

        # Convert transient class codes into names
        class_names = [code_cat[cc] for cc in class_accuracies.keys()]

        # Generate array of length tclasses - class names will be assigned below
        # in yticks , in same order as these indices from tclasses_names
        class_index = np.arange(len(class_accuracies.keys()))

        # Plot and save figure
        rcParams['figure.figsize'] = 8, 8
        accuracies = [class_accuracies[c] for c in class_accuracies.keys()]
        plt.bar(class_index, accuracies)

        cur_class = 0
        for xy in zip(class_index, accuracies):
            cur_count = class_counts[cur_class]  # Get class count of current class_index
            cur_class += 1
            plt.annotate(str(cur_count) + " total", xy=xy, textcoords='data',
                         ha='center', va='bottom', fontsize=12)
        plt.xlabel('Transient Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(class_index, class_names, fontsize=12, rotation=30)
        plt.yticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=12)
        plot_title = self.name + " Accuracy, on Testing Data"
        plt.title(plot_title, fontsize=15)
        plt.show()
