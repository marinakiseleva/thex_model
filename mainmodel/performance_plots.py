
"""
Class Mixin for MainModel which contains all performance plotting functionality 
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import utilities.utilities as thex_utils


from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI, TREE_ROOT, UNDEF_CLASS


class MainModelVisualization:
    """
    Mixin Class for MainModel performance visualization
    """

    def compute_performance(self, class_metrics):
        """
        Get recall, precision, and specificities per class. 
        """
        precisions = {}
        recalls = {}
        specificities = {}  # specificity = true negative rate
        for class_name in class_metrics.keys():
            metrics = class_metrics[class_name]
            den = metrics["TP"] + metrics["FP"]
            precisions[class_name] = metrics["TP"] / den if den > 0 else 0
            den = metrics["TP"] + metrics["FN"]
            recalls[class_name] = metrics["TP"] / den if den > 0 else 0

            specificities[class_name] = metrics["TN"] / \
                (metrics["TN"] + metrics["FP"])
        return recalls, precisions, specificities

    def plot_all_metrics(self, class_metrics, y):
        """
        Plot performance metrics for model
        :param class_metrics: Returned from compute_metrics; Map from class name to map of performance metrics
        """
        class_counts = self.get_class_counts(y)
        recalls, precisions, specificities = self.compute_performance(class_metrics)
        pos_baselines, neg_baselines, precision_baselines = self.compute_baselines(
            class_counts, y)

        self.plot_metrics(recalls, "Completeness", pos_baselines)
        self.plot_metrics(
            specificities, "Completeness of Negative Class Presence", neg_baselines)
        self.plot_metrics(precisions, "Purity", precision_baselines)

    def plot_metrics(self, class_metrics, xlabel, baselines=None):
        """
        Visualizes accuracy per class with bar graph; with random baseline based on class level in hierarchy.
        :param class_metrics: Mapping from class name to metric value.
        :param xlabel: Label to assign to y-axis
        :[optional] param baselines: Mapping from class name to random-baseline performance
        # of samples in each class (get plotted atop the bars)
        :[optional] param annotations: List of
        """
        class_names, metrics, baselines = self.get_ordered_metrics(
            class_metrics, baselines)
        # Set constants
        tick_size = 8
        bar_width = 0.8
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, tight_layout=True)
        ax = plt.subplot()

        y_indices = np.arange(len(metrics))

        # Plot metrics
        ax.barh(y=y_indices, width=metrics, height=bar_width)

        # Plot random baselines
        if baselines is not None:
            for index, baseline in enumerate(baselines):
                plt.vlines(x=baseline, ymin=index - (bar_width / 2),
                           ymax=index + (bar_width / 2), linestyles='--', colors='red')

        # Format Axes, Labels, and Ticks
        ax.set_xlim(0, 1)
        plt.xticks(list(np.linspace(0, 1, 11)), [
                   str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
        plt.xlabel(xlabel, fontsize=10)
        plt.yticks(y_indices, class_names,  fontsize='xx-small',
                   horizontalalignment='left')
        plt.ylabel('Transient Class', fontsize=10)
        max_tick_width = 0
        for i in class_names:
            bb = mpl.textpath.TextPath((0, 0), i, size=tick_size).get_extents()
            max_tick_width = max(bb.width, max_tick_width)
        yax = ax.get_yaxis()
        yax.set_tick_params(pad=max_tick_width + 2)

        thex_utils.display_and_save_plot(self.name, xlabel, ax)

    def get_ordered_metrics(self, class_metrics, baselines=None):
        """
        Reorder metrics and reformat class names in hierarchy groupings
        :param class_metrics: Mapping from class name to metric value.
        :[optional] param baselines: Mapping from class name to random-baseline performance
        """
        ordered_names = sorted(self.class_labels)
        ordered_formatted_names = []
        ordered_metrics = []
        ordered_baselines = [] if baselines is not None else None
        for class_name in ordered_names:
            # Add to metrics and baselines
            ordered_metrics.append(class_metrics[class_name])
            if baselines is not None:
                ordered_baselines.append(baselines[class_name])

            # Reformat class name based on depth and add to names
            pretty_class_name = class_name
            if UNDEF_CLASS in class_name:
                pretty_class_name = class_name.replace(UNDEF_CLASS, "")
                pretty_class_name = pretty_class_name + " (unspec.)"
            # level = self.class_levels[class_name]
            # if level > 2:
            #     indent = " " * (level - 1)
            #     symbol = ">"
            #     pretty_class_name = indent + symbol + pretty_class_name
            ordered_formatted_names.append(pretty_class_name)

        ordered_formatted_names.reverse()
        ordered_metrics.reverse()
        if baselines is not None:
            ordered_baselines.reverse()
        return [ordered_formatted_names, ordered_metrics, ordered_baselines]

    def plot_probability_vs_accuracy(self, range_metrics):
        """
        Plots accuracy of class (y-axis) vs. probability assigned to class (x-axis). Accuracy is measured as the number of true positives (samples with this class as label) divided by all samples with probabilities assigned in this range. 
        :param range_metrics: Map of classes to [TP_range_sums, total_range_sums]
            total_range_sums: # of samples with probability in range for this class
            TP_range_sums: true positives per range 

        """

        for index, class_name in enumerate(range_metrics.keys()):
            true_positives, totals = range_metrics[class_name]
            f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

            accuracies = []  # Accuracy per range (true positive/total)
            for index, TP in enumerate(true_positives):
                p = 0
                total_predicted = totals[index]
                if total_predicted != 0:
                    # positive class samples / totals # with prob in range
                    p = TP / total_predicted
                accuracies.append(p)

            normalize = plt.Normalize(min(true_positives), max(true_positives))
            colors = plt.cm.Blues(normalize(true_positives))

            perc_ranges = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

            x_indices = np.arange(len(accuracies))
            ax.bar(x_indices, accuracies)  # color=bar_colors

            thex_utils.annotate_plot(ax, x_indices, accuracies, totals)

            plt.yticks(list(np.linspace(0, 1, 11)), [
                       str(tick) + "%" for tick in list(range(0, 110, 10))], fontsize=10)
            plt.xticks(x_indices, perc_ranges, fontsize=10)
            plt.xlabel('Probability of ' + class_name + ' +/- 5%', fontsize=12)
            plt.ylabel('Purity', fontsize=12)
            thex_utils.display_and_save_plot(self.name,
                                             "Probability vs Positive Rate: " + class_name, ax)
