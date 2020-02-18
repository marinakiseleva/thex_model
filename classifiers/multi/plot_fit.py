import numpy as np
import matplotlib.pyplot as plt

from thex_data.data_consts import TARGET_LABEL, ROOT_DIR, FIG_WIDTH, FIG_HEIGHT, DPI
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
    title = class_name + " : " + feature_name
    plt.title(title, fontsize=12)
    plt.xlabel(feature_name, fontsize=10)
    plt.ylabel("Density", fontsize=10)
    # plt.xlim(left=0, right=max_value)

    # sample probabilities for a range of outcomes
    values = np.linspace(-3, 2, 100)
    values = values.reshape(len(values), 1)

    probabilities = kde.score_samples(values)
    probabilities = np.exp(probabilities)
    # plot the histogram and pdf
    ax.hist(data, bins=10, density=True)
    ax.plot(values, probabilities)
    util.display_and_save_plot(model_dir, "fit_" + title, ax)
