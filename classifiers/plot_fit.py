import numpy as np
import matplotlib.pyplot as plt

from thex_data.data_consts import FIG_WIDTH, FIG_HEIGHT, DPI
import utilities.utilities as util


def plot_fit(data, kde, feature_name, class_name, model_dir):
    """
    Overlay data with KDE to see how well KDE fits data.
    :param data: 2D numpy array of single feature
    :param kde: KDE fit to data
    :param feature_name: name of feature of the data
    :param class_name: name of class of the data
    :param model_dir: Model directory to save to
    """
    f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    ax.set_title('\n'.join(wrap(class_name + " : " + feature_name, 60)))
    plt.xlabel(feature_name, fontsize=10)
    plt.ylabel("Density", fontsize=10)

    values = np.linspace(-3, 2, 100)
    values = values.reshape(len(values), 1)

    # sample probabilities for the range of values
    probabilities = np.exp(kde.score_samples(values))
    # plot the histogram and pdf
    ax.hist(data, bins=10, density=True)
    ax.plot(values, probabilities)
    util.save_plot(model_dir, class_name + " : " + feature_name)


def plot_fits(kdes, features, classes, model_dir):
    """
    Plot KDEs of the same feature and different classes together, so to compare different classes.
    :param kdes: Map from class name to map from features to KDEs
    :param features: All features to iterate through
    :param class_name: All classes to iterate through
    :param model_dir: Model directory to save to
    """
    values = np.linspace(-3, 2, 100)
    values = values.reshape(len(values), 1)

    for feature_name in features:
        f, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
        ax.set_title(feature_name)
        plt.xlabel(feature_name, fontsize=10)
        plt.ylabel("Probability Density", fontsize=10)

        # Plot KDE fit for this feature, per class
        for class_name in classes:
            kde = kdes[class_name][feature_name]
            if kde is not None:
                # sample probabilities for the range of values
                probabilities = np.exp(kde.score_samples(values))
                ax.plot(values, probabilities, label=class_name)
        ax.legend()

        util.save_plot(model_dir, feature_name)
