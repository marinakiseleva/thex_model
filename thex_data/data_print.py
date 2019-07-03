"""
data_print
Print useful information about data
"""
import pandas as pd

from .data_consts import code_cat, TARGET_LABEL


def print_class_counts(df):
    """
    Prints number of data points in each class
    """
    classes = df[TARGET_LABEL].value_counts().keys().tolist()
    class_names = [code_cat[code] for code in classes]
    class_counts = df[TARGET_LABEL].value_counts().tolist()
    print("\nCounts per Class\n------------------")
    for index, value in enumerate(class_names):
        print(str(value) + " : " + str(class_counts[index]))
    print("\nTotal: " + str(df.shape[0]))


def print_styled(input_data, input_name):
    print("\n" + input_name + "\n" + "------------------")
    print(input_data)


def print_priors(priors):
    print("\nPriors\n------------------")
    for k in priors.keys():
        print(code_cat[k] + " : " + str(priors[k]))


def print_features_used(col_list):
    print("\nFeatures\n------------------")
    print("Total: " + str(len(col_list)))
    out_list = col_list[0]
    for c in col_list[1:]:
        out_list += ", " + c
    print(out_list)


def print_filters(filters):
    print("\nData Filters\n------------------")
    for data_filter in filters.keys():
        print(str(data_filter) + " : " + str(filters[data_filter]))
