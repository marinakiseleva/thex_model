import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) +
                "/../../libraries/hmc/hmc")

import hmc
import metrics
from datasets import *

from thex_data.data_prep import get_train_test
from thex_data.data_consts import class_to_subclass as hierarchy
from thex_data.data_consts import TARGET_LABEL, code_cat, cat_code
from model_performance.performance import *


def run_tree(data_columns, incl_redshift):
    X_train, X_test, y_train, y_test = get_train_test(
        data_columns, incl_redshift)

    # Convert codes back to category titles since hierarchy is in category form
    y_train_strs, y_test_strs = convert_target(y_train, y_test)

    tree, hmc_hierarchy = train_tree(X_train, y_train_strs)
    evaluate_tree(tree, hmc_hierarchy, X_test, y_test_strs)


def convert_target(y_train, y_test):
    """
    Converts y train and test of transient codes to their category names for the tree (because the hierarchy is defined in terms of names, not codes)
    """
    y_train_strs = y_train.copy()
    y_test_strs = y_test.copy()
    y_train_strs[TARGET_LABEL] = y_train_strs[TARGET_LABEL].apply(lambda x: code_cat[x])
    y_test_strs[TARGET_LABEL] = y_test_strs[TARGET_LABEL].apply(lambda x: code_cat[x])
    return y_train_strs, y_test_strs


def train_tree(X, y):
    """
    Builds hierarchy and tree, and fits X, y data to them. 
    :param X: pandas dataframe of features
    :param y: pandas dataframe of target in category name form
    """
    hmc_hierarchy = init_hierarchy()
    hmc_hierarchy.print_()
    tree = hmc.DecisionTreeHierarchicalClassifier(hmc_hierarchy)
    tree = tree.fit(X, y)

    return tree, hmc_hierarchy


def evaluate_tree(tree, hmc_hierarchy, X_test, y_test):
    predictions = tree.predict(X_test)
    dth_accuracy = tree.score(X_test, y_test)

    print("Accuracy: %s" % metrics.accuracy_score(hmc_hierarchy, y_test, predictions))
    print("-------------------------- Ancestors -------------------------- ")
    print("Precision Ancestors: %s" %
          metrics.precision_score_ancestors(hmc_hierarchy, y_test, predictions))
    print("Recall Ancestors: %s" % metrics.recall_score_ancestors(
        hmc_hierarchy, y_test, predictions))
    print("F1 Ancestors: %s" % metrics.f1_score_ancestors(
        hmc_hierarchy, y_test, predictions))
    print("-------------------------- Descendants -------------------------- ")
    print("Precision Descendants: %s" %
          metrics.precision_score_descendants(hmc_hierarchy, y_test, predictions))
    print("Recall Descendants: %s" % metrics.recall_score_descendants(
        hmc_hierarchy, y_test, predictions))
    print("F1 Descendants: %s" % metrics.f1_score_descendants(
        hmc_hierarchy, y_test, predictions))

    prediction_codes = [cat_code[cc] for cc in predictions]
    actual_codes = [cat_code[cc] for cc in y_test[TARGET_LABEL].values]
    plot_confusion_matrix(actual_codes, prediction_codes,
                          normalize=True, title='Confusion matrix')

    compute_plot_class_accuracy(prediction_codes, actual_codes, "HMC Tree Accuracy")


def init_hierarchy():
    # TTypes  = top-level class in data_consts map.
    hmc_hierarchy = hmc.ClassHierarchy("TTypes")
    for parent in hierarchy.keys():
        # hierarchy maps parents to children, so get all children
        list_children = hierarchy[parent]
        for child in list_children:
            # Nodes are added with child parent pairs
            try:
                hmc_hierarchy.add_node(child, parent)
            except ValueError as e:
                print(e)

    return hmc_hierarchy
