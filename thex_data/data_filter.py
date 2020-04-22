"""
data_filter
Functionality for filtering data
"""

import pandas as pd

from .data_consts import TARGET_LABEL
import utilities.utilities as util


def drop_disjoint_conflicts(X, y, class_labels):
    """
    Drop rows that have more than 1 disjoint label assigned (conflicting labels)
    """
    keep_indices = []
    for index, row in y.iterrows():
        keep = True
        labels = util.convert_str_to_list(row[TARGET_LABEL])

        i = list(set(labels).intersection(set(class_labels)))
        if len(i) == 1:
            # Contains 1 disjoint label
            keep_indices.append(index)

    k = list(set(keep_indices))
    return X.loc[k, :].reset_index(drop=True), y.loc[k, :].reset_index(drop=True)


def filter_data(X, y, data_filters, class_labels, class_hier):
    """
    Filter data with known class labels and default/passed-in filters
    """
    min_class_size = data_filters['min_class_size']
    max_class_size = data_filters['max_class_size']

    # Filter data (keep only classes that are > min class size)
    data = pd.concat([X, y], axis=1)

    # Keeps only samples whose class is in classes and has the minimum number
    # of class samples necessary.
    filtered_data = filter_class_size(data,
                                      min_class_size,
                                      class_labels)

    if max_class_size is not None:
          # Randomly subsample any over-represented classes down to passed-in value
        filtered_data = sub_sample(filtered_data,
                                   max_class_size,
                                   class_labels)

    X = filtered_data.drop([TARGET_LABEL], axis=1).reset_index(drop=True)
    y = filtered_data[[TARGET_LABEL]].reset_index(drop=True)

    X, y = drop_disjoint_conflicts(X, y, class_labels)

    return X, y


def drop_conflicts(df):
    """
    Drop rows where classes belong to classes in different disjoint groups. This requires special handling not yet implemented.
    """
    conflict_labels = ["_CONFUSION", "_CONFLICT",
                       "__CONFLICT_CASES", "_UNCLEAR_LABELS", "_IGNORED_LABELS"]
    keep_indices = []
    for df_index, row in df.iterrows():
        list_classes = util.convert_str_to_list(row[TARGET_LABEL])
        keep = True
        for c in list_classes:
            if c in conflict_labels:
                keep = False
        if keep:
            keep_indices.append(df_index)

    return df.loc[keep_indices, :]


def infer_data_classes(df):
    """
    Helper function to infer unique classes from data
    """
    unique_class_groups = list(df[TARGET_LABEL].unique())
    unique_classes = set()
    for group in unique_class_groups:
        for class_name in util.convert_str_to_list(group):
            unique_classes.add(class_name)
    return list(unique_classes)


def filter_columns(df, col_list):
    """
    Filters columns down to those passed in as col_list (+ target label)
    """
    # Filter DataFrame down to these columns
    col_list = col_list + [TARGET_LABEL]
    df = df[col_list]
    return df


def sub_sample(df, count, classes):
    """
    Sub-samples over-represented class
    :param df: DataFrame to subsample count, if class has less than or equal to this count, then just leave it
    :param count: Randomly subsample all classes to this
    :param classes: classes to filter on.
    """
    if count is None:
        return df

    if classes is None:
        # Infer unique classes from data
        classes = infer_data_classes(df)

    # Filter each class in list, by indices. Save indices of samples to keep.
    keep_indices = []
    for class_label in classes:
        class_indices = []
        for index, row in df.iterrows():
            class_list = util.convert_str_to_list(row[TARGET_LABEL])
            if class_label in class_list:
                class_indices.append(index)
        df_class = df.loc[class_indices, :]
        num_rows = df_class.shape[0]  # number of rows

        if num_rows > count:
            # Reduce to the count number
            sampled_class_indices = list(df_class.sample(n=count).index)
            keep_indices += sampled_class_indices
        else:
            keep_indices += class_indices

    unique_indices = list(set(keep_indices))

    return df.loc[unique_indices, :].reset_index(drop=True)


def filter_class_size(df, N, classes):
    """
    Keep samples that have a class in classes with at least N points
    :param df: DataFrame of features and TARGET_LABEL
    :param N: Minimum number of samples required in class to keep it
    :return: DataFrame of original format, with only top N classes of data
    """
    if N is None:
        return df

    if classes is None:
        raise ValueError("Need class labels.")

    # Filter each class in list, by indices. Save indices of samples to keep.
    keep_indices = []
    for class_label in classes:
        class_indices = []
        for index, row in df.iterrows():
            class_list = util.convert_str_to_list(row[TARGET_LABEL])
            if class_label in class_list:
                class_indices.append(index)
        df_class = df.loc[class_indices, :]
        num_rows = df_class.shape[0]  # number of rows

        if num_rows >= N:
            keep_indices += class_indices

    unique_indices = list(set(keep_indices))

    return df.loc[unique_indices, :].reset_index(drop=True)
