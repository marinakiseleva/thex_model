import argparse
import pandas as pd
import sys
# sys.path.append("..")
from thex_data.data_prep import init_data
from thex_data import data_plot
import nb_performance as nbp
from nb_classifier import *


"""
Runs Gaussian Naive Bayes Classifier
"""


def cross_validate(data_columns, incl_redshift=False, k=5):
    """
    Split data into k-folds and perform k-fold cross validation
    """

    data = get_data(data_columns, incl_redshift)


def run_model(data_columns, incl_redshift=False, plot_title="Naive Bayes Accuracy per Class"):
    """
    Main runner for Naive Bayes model
    :param data_columns: list of specific column names to use
    :param incl_redshift: will include redshift as a variable for training
    :param plot_title: Title of accuracy plots created, useful to write here what set of data columns were used

    """
    train, test = init_data(col_list=data_columns, incl_redshift=incl_redshift)

    # Initialize Naive Bayes classifier with training data
    summaries, priors = summarize_by_class(train)

    # Predict on test data without using transient type
    test_set = test.drop(["transient_type"], axis=1)

    predictions = test_set_samples(summaries, priors, test_set)
    predicted_classes = pd.DataFrame(predictions, columns=['predicted_class'])
    actual_classes = test['transient_type'].reset_index(drop=True)

    nbp.get_accuracy(predicted_classes, actual_classes)

    nbp.plot_class_accuracy(predicted_classes, actual_classes, plot_title)

    # data_plot.plot_feature_distribution(train)
    data_plot.plot_class_distribution(train)
    return predicted_classes, actual_classes


def compare_datasets(data_columns, plot_title="Naive Bayes Accuracy per Class, compared"):
    """
    Compare 2 different datasets and plot accuracy outputs on comparison bar graph
    """
    predicted_classes, actual_classes = run_model(data_columns)
    predicted_classes2, actual_classes2 = run_model(
        data_columns=data_columns, incl_redshift=True)

    # comparing non redshift to redshift valued
    nbp.plot_compare_accuracy(predicted_classes, actual_classes,
                              predicted_classes2, actual_classes2,
                              "Without redshift", "With redshift", plot_title)

from thex_model.data_init import collect_data
import os


def main():
    parser = argparse.ArgumentParser(description='Classify transients')
    # Pass in columns as space-delimited texts, like this:
    # PS1_gKmag PS1_rKmag PS1_iKmag PS1_zKmag PS1_yKmag
    parser.add_argument('-cols', '--cols', nargs='+',
                        help='<Required> Pass column names', required=False)
    # parser.add_argument('-col_names', '--col_names', nargs='+',
    # help='Pass in string by which columns will be selected. For example:
    # AllWISE will use all AlLWISE columns.', required=False)
    parser.add_argument('-incl_redshift', '--incl_redshift', nargs='+',
                        help='<Required> Set flag', required=False)
    args = parser.parse_args()

    # if args.col_names is None and args.cols is None:
    #     print("Either col_name or cols needs to be passed in. Exiting.")
    #     return -1

    # col_list = []
    # if args.col_names is not None:
    #     # Make list of column/feature names; exlcude _e_ (errors)
    #     col_list = [col for col in list(df) if any(
    #         col_val in col and "_e_" not in col for col_val in args.col_names)]
    # elif args.cols is not None:
    col_list = args.cols
    incl_redshift = args.incl_redshift if args.incl_redshift is not None else False
    # compare_datasets(col_list)
    run_model(data_columns=col_list, incl_redshift=incl_redshift)

if __name__ == '__main__':
    main()
