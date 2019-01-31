
# import pandas as pd
from sklearn.model_selection import KFold

from thex_data.data_consts import code_cat, TARGET_LABEL
from thex_data.data_prep import get_train_test
from thex_data import data_plot
from model_performance.init_classifier import *
from model_performance.performance import *
from nb_performance import *
from nb_train import *
from nb_test import *


"""
Naive Bayes Classifier
"""


def run_model(data_columns, incl_redshift=False, plots=True):
    """
    Main runner for Naive Bayes model
    :param data_columns: list of specific column names to use
    :param incl_redshift: will include redshift as a variable for training
    """

    X_train, X_test, y_train, y_test = get_train_test(
        data_columns, incl_redshift=incl_redshift)
    # Train: Initialize Naive Bayes classifier with training data
    summaries, priors = train_nb(X_train, y_train)

    # Test: Predict on test data without using transient type
    predicted_classes = test_model(X_test, summaries, priors)
    print(predicted_classes)

    if plots:
        total_accuracy = get_accuracy(predicted_classes, y_test)
        compute_plot_class_accuracy(
            predicted_classes, y_test, plot_title="Naive Bayes Accuracy, on Testing Data")

        # get_rocs(test, summaries, priors)

        # for f in list(train):
        #     data_plot.plot_feature_distribution(train, feature=f)

        plot_confusion_matrix(y_test, predicted_classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues)
    return predicted_classes, y_test


def compare_datasets(data_columns, plot_title="Naive Bayes Accuracy per Class, compared"):
    """
    Run model with and without redshift and compare accuracies
    """
    predicted_classes, actual_classes = run_model(data_columns)
    predicted_classes2, actual_classes2 = run_model(
        data_columns=data_columns, incl_redshift=True)

    # comparing non redshift to redshift valued
    plot_compare_accuracy(predicted_classes, actual_classes,
                          predicted_classes2, actual_classes2,
                          "Without redshift", "With redshift", plot_title)


def main():
    col_list, incl_redshift = collect_args()
    run_model(data_columns=col_list, incl_redshift=incl_redshift)
    # run_cv(data_columns=col_list, incl_redshift=incl_redshift)


if __name__ == '__main__':
    main()
