"""
data_clean:
Reassigns labels for data. convert_class_vectors creates a vector of Booleans for class presence like [0,0,0,0,1,0,1, ...], and remaining functionality is predominantly for assigning each sample a single class. 

"""
import numpy as np
import pandas as pd

from hmc import hmc

from .data_consts import groupings, ORIG_TARGET_LABEL, TARGET_LABEL, cat_code, UNDEF_CLASS
from .data_consts import class_to_subclass as full_class_hierarchy


def relabel(class_index, class_vectors):
    """
    Relabel samples such that if they have a 1 in their class vector for class_index, they will be relabeled as 1; otherwise 0. Relabels TARGET_LABEL column of class_vectors
    :return: Pandas DataFrame with TARGET_LABEL column, filling with 1 if  class_vectors[TARGET_LABEL][class_index] is also 1, otherwise 0
    """
    labels = []
    for df_index, row in class_vectors.iterrows():
        class_vector = row[TARGET_LABEL]
        p = 1 if class_vector[class_index] == 1 else 0
        labels.append(p)
    class_vectors = pd.DataFrame(labels, columns=[TARGET_LABEL])
    return class_vectors


def convert_class_vectors(df, class_labels, level=None):
    """
    Convert labels of TARGET_LABEL column in passed-in DataFrame to class vectors. If level is passed, class is assigned to the parent's Undefined group if it does not have a subclass. For example Ia becomes Undefined_Ia if the sample is not a subtype of Ia. 
    :param class_labels: Class names to consider. 
    :param level: self.test_level; If sample does not have a class on this level, it is assigned the parent undefined class. 
    :return class_vectors: DataFrame with same number of rows as df, with only TARGET_LABEL column. Each row has a single vector in that columns, with the same length as class_labels, and where values are 0 or 1. 1 if it is that class, 0 otherwise.

    """
    # Convert labels to class vectors, with 1 meaning it has that class, and 0
    # does not
    rows_list = []
    for df_index, row in df.iterrows():
        class_vector = [0] * len(class_labels)
        cur_classes = convert_str_to_list(row[TARGET_LABEL])
        has_level_class = False
        for class_index, c in enumerate(class_labels):
            if c in cur_classes:
                class_vector[class_index] = 1
                has_level_class = True
        if level is not None and has_level_class == False:
            # Assign parent's undefined class .
            for class_index, c in enumerate(class_labels):
                if UNDEF_CLASS in c and c[len(UNDEF_CLASS):] in cur_classes:
                    class_vector[class_index] = 1

        rows_list.append([class_vector])
    class_vectors = pd.DataFrame(rows_list, columns=[TARGET_LABEL])
    return class_vectors


def init_tree(class_hierarchy=None):
    print("\n\nConstructing Class Hierarchy Tree...")
    hmc_hierarchy = hmc.ClassHierarchy("TTypes")
    hierarchy = full_class_hierarchy if class_hierarchy is None else class_hierarchy
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


def assign_levels(tree, mapping, node, level):
    """
    Assigns level to each node based on level in hierarchical tree. The lower it is in the tree, the larger the level. The level at the root is 1. Returns mapping of class name to level.
    """
    mapping[str(node)] = level
    for child in tree._get_children(node):
        assign_levels(tree, mapping, child, level + 1)
    return mapping


def convert_str_to_list(input_string):
    """
    Convert string to list
    """
    l = input_string.split(",")
    return [item.strip(' ') for item in l]


def group_by_tree(df, transform_labels):
    """
    Normalized claimed type (transient type). If claimed type is not in map, it is removed. Only considers 1-1 mappings, does not use galaxies that have more than 1. Defines new TARGET_LABEL column. 
    :param df: DataFrame of values and labels. Must have column ORIG_TARGET_LABEL
    :return df: Returns Pandas DataFrame with new column TARGET_LABEL, which has normalized transient type for each 
    """
    if transform_labels == False:
        df[TARGET_LABEL] = df[ORIG_TARGET_LABEL]
        df = df[~df[TARGET_LABEL].isnull()]
        return df

    tree = init_tree()
    node_depths = assign_levels(tree, {}, tree.root, 1)

    for index, row in df.iterrows():
        orig_labels = convert_str_to_list(row[ORIG_TARGET_LABEL])

        label = find_max_label(orig_labels, node_depths)
        # label = find_min_label(orig_labels, node_depths)

        df.at[index, ORIG_TARGET_LABEL] = label

    df[TARGET_LABEL] = df[ORIG_TARGET_LABEL].apply(
        lambda x: groupings[x] if x in groupings else None)
    df = df[~df[TARGET_LABEL].isnull()]

    # Convert transient class to number code
    df[TARGET_LABEL] = df[TARGET_LABEL].apply(lambda x: int(cat_code[x]))
    return df


def find_min_label(labels, node_depths):
    """
    Find label with minimal depth (highest in the tree, excluding root), ie. I, II, CC, etc.
    """
    min_depth = 10
    min_label = None
    for label in labels:
        if label in groupings:
            # Groups labels into mapped definitions, ie. 1c -> Ic
            label = groupings[label]

        if label in node_depths and node_depths[label] < min_depth and label != 'TTypes':
            min_depth = node_depths[label]
            min_label = label
    return min_label


def find_max_label(labels, node_depths):
    """
    Find label with maximal depth (deepest in the tree), ie. Ia Pec, IIb, etc.
    """
    max_depth = 0
    max_label = None

    for label in labels:
        if label in groupings:
            # Groups labels into mapped definitions, ie. 1c -> Ic
            label = groupings[label]

        if label in node_depths and node_depths[label] > max_depth:
            max_depth = node_depths[label]
            max_label = label
    return max_label


def fill_nulls(df):
    """
    Fills NULL values in dataframe by assigning average value in the column 
    """
    for col_name in list(df):
        df[col_name].fillna((df[col_name].mean()), inplace=True)

        mask = df[col_name] != np.inf
        df.loc[~mask, col_name] = df.loc[mask, col_name].mean()

    return df
