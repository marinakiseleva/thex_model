import argparse
import pandas as pd
import sys
from sklearn.model_selection import KFold

from thex_data.data_consts import code_cat, TARGET_LABEL
from thex_data.data_prep import get_train_test, get_data
from thex_data import data_plot
from thex_data.data_init import *

from model_performance.performance import *

from nb_performance import *
from nb_train import *
from nb_test import *


"""
Runs Naive Bayes Classifier
"""


def run_cv(data_columns, incl_redshift=False, k=3):
    """
    Split data into k-folds and perform k-fold cross validation
    """

    data = get_data(data_columns, incl_redshift)
    print("Rows in dataset: " + str(data.shape[0]))

    kf = KFold(n_splits=k, shuffle=True, random_state=10)
    kf.get_n_splits(data)
    accuracies = []
    accuracy_per_class = {}
    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Train
        summaries, priors = summarize_by_class(train_data)

        # Test
        predicted_classes, actual_classes = test_model(
            test_data, summaries, priors)

        df_compare = combine_dfs(predicted_classes, actual_classes)
        # Get accuracy per class of transient
        class_accuracies, classes = get_class_accuracies(df_compare)
        for index, c in enumerate(classes):
            if c in accuracy_per_class:
                accuracy_per_class[c] += class_accuracies[index]
            else:
                accuracy_per_class[c] = class_accuracies[index]

        total_accuracy = get_accuracy(predicted_classes, actual_classes)
        accuracies.append(total_accuracy)

    accs = {code_cat[c]: acc / k for c, acc in accuracy_per_class.items()}

    plot_class_accuracy(list(accs.values()), list(
        accs.keys()), "CV Accuracy with " + str(k) + " folds")

    avg_accuracy = sum(accuracies) / k
    print("Average accuracy: " + str(avg_accuracy))

    # for f in list(data):
    #     data_plot.plot_feature_distribution(data, feature=f)
    return avg_accuracy


def run_model(data_columns, incl_redshift=False):
    """
    Main runner for Naive Bayes model
    :param data_columns: list of specific column names to use
    :param incl_redshift: will include redshift as a variable for training
    """
    train, test = get_train_test(col_list=data_columns, incl_redshift=incl_redshift)

    # Train: Initialize Naive Bayes classifier with training data
    summaries, priors = summarize_by_class(train)

    # print("Testing model on Training data...")
    # test = train.copy()

    # Test: Predict on test data without using transient type
    predicted_classes, actual_classes = test_model(
        test, summaries, priors)

    total_accuracy = get_accuracy(predicted_classes, actual_classes)
    compute_plot_class_accuracy(
        predicted_classes, actual_classes, plot_title="Naive Bayes Accuracy, on Testing Data")

    # get_rocs(test, summaries, priors)

    # for f in list(train):
    #     data_plot.plot_feature_distribution(train, feature=f)

    plot_confusion_matrix(actual_classes, predicted_classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)


def test_model(test, summaries, priors):
    """
    Tests model using established summaires & priors on test data (includes ttype)
    """
    test_set = test.drop([TARGET_LABEL], axis=1)
    predictions = test_set_samples(summaries, priors, test_set)
    predicted_classes = pd.DataFrame(predictions, columns=['predicted_class'])
    actual_classes = test[TARGET_LABEL].reset_index(drop=True)

    return predicted_classes, actual_classes


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
    parser = argparse.ArgumentParser(description='Classify transients')
    # Pass in columns as space-delimited texts, like this:
    # PS1_gKmag PS1_rKmag PS1_iKmag PS1_zKmag PS1_yKmag
    parser.add_argument('-cols', '--cols', nargs='+',
                        help='<Required> Pass column names', required=False)
    parser.add_argument('-col_names', '--col_names', nargs='+',
                        help='Pass in string by which columns will be selected. For example: AllWISE will use all AlLWISE columns.', required=False)
    parser.add_argument('-incl_redshift', '--incl_redshift', nargs='+',
                        help='<Required> Set flag', required=False)
    args = parser.parse_args()

    col_list = []
    if args.cols is None:
        if args.col_names is not None:
            col_list = collect_cols(
                "./../../../data/THEx-catalog.v0_0_3.fits", args.col_names)
        else:
            print("cols or col_names needs to be passed in. Exiting.")
            return -1
    else:
        col_list = args.cols
    incl_redshift = args.incl_redshift if args.incl_redshift is not None else False
    print("\n\n################################################################################# Using col_list\n" + str(col_list))
    # compare_datasets(col_list)
    run_model(data_columns=col_list, incl_redshift=incl_redshift)
    # run_cv(data_columns=col_list, incl_redshift=incl_redshift)


# def get_best_col_set(data_df):
#     col_lists = {
#         'ned_galex_2mass': ['NED_GALEX_FUV', 'NED_GALEX_NUV', 'NED_2MASS_J', 'NED_2MASS_H', 'NED_2MASS_Ks'],
#         'ned_sdss': ['NED_SDSS_u', 'NED_SDSS_g', 'NED_SDSS_r', 'NED_SDSS_i', 'NED_SDSS_z']
#         'hyperleda' = [col for col in list(data_df) if "HyperLEDA" in col],
#         'ned_irac_iras': [col for col in list(data_df) if ("IRAC_" in col or 'IRAS_' in col) and 'Err' not in col],
#         'allwise': [col for col in list(data_df) if "AllWISE" in col and "Err" not in col],
#         'firefly_cols': [col for col in list(data_df) if "Firefly" in col],  # Firefly
#         'mpa_cols': [col for col in list(data_df) if "MPAJHU" in col],  # MPAJHU
#         'zoo_cols': [col for col in list(data_df) if "Zoo" in col],  # GalaxyZoo
#         'gswlc': [col for col in list(data_df) if "GSWLC" in col],  # GSWLC
#         'wiscpca': [col for col in list(data_df) if "WiscPCA" in col],  # WiscPCA
#         'nsa': [col for col in list(data_df) if "NSA_" in col],  # NSA
#         # NSA
#         'nsa_k': [col for col in list(data_df) if "NSA_" in col and "KCORRECT_" in col],
#         'scos': [col for col in list(data_df) if "SCOS_" in col],  # SCOS
#         'ps1': [col for col in list(data_df) if "PS1" in col and '_e_' not in col]  # PS1
#         # 'full' : scos + ps1 + nsa + wiscpca + gswlc + zoo_cols + mpa_cols
#     }


if __name__ == '__main__':
    main()
