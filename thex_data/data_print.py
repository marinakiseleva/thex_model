import pandas as pd

from .data_consts import code_cat, TARGET_LABEL


"""
Print useful information about data
"""


def get_class_counts(df):
    """
    Prints number of data points in each class
    """

    classes = df[TARGET_LABEL].value_counts().keys().tolist()
    class_names = [code_cat[code] for code in classes]
    class_counts = df[TARGET_LABEL].value_counts().tolist()
    print("Counts per Class\n------------------ ")
    for index, value in enumerate(class_names):
        print(str(value) + " : " + str(class_counts[index]))
