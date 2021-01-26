"""
data_transform
Enhance features by scaling and transforming them
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from thex_data.data_consts import ORDERED_MAGS


def transform_features(df):
    """
    Computes feature transformations. Enhances existing features by computing new representations of same features.
    :param df: DataFrame of features
    """
    df = derive_diffs(df)
    return df


def derive_diffs(df):
    """
    Subtracts adjacent color bands from one another and saves result in new columns, labeled as differences between two
    :param df: DataFrame of features
    """
    features = list(df)
    for index, colname1 in enumerate(features):
        if colname1 in ORDERED_MAGS:
            colname2 = ORDERED_MAGS[colname1]
            if colname2 in features:
                primary_mag = df[colname1]
                prev_mag = df[colname2]
                new_col_name = colname2 + "_minus_" + colname1
                df[new_col_name] = prev_mag - primary_mag

    return df


def scale_data(X_train, X_test, X_val=None):
    """
    Fit scaling to training data and apply to both training and testing; scale by removing mean and scaling to unit variance. 
    Returns X_train and X_test as Pandas DataFrames
    :param X_train: Pandas DataFrame of training data
    :param X_test: Pandas DataFrame of testing data
    """
    features_list = list(X_train)
    scaler = StandardScaler()

    scaled_X_train = pd.DataFrame(
        data=scaler.fit_transform(X_train), columns=features_list)
    scaled_X_test = pd.DataFrame(
        data=scaler.transform(X_test), columns=features_list)

    if X_val is not None:
        scaled_X_val = pd.DataFrame(
            data=scaler.transform(X_val), columns=features_list)
        return scaled_X_train, scaled_X_test, scaled_X_val

    return scaled_X_train, scaled_X_test


def apply_PCA(self, X_train, X_test, k):
    """
    Fit PCA to training data and apply to both training and testing;
    Returns X_train and X_test as Pandas DataFrames
    :param X_train: Pandas DataFrame of training data
    :param X_test: Pandas DataFrame of testing data
    :param k: self.pca from model; number of components
    """
    def convert_to_df(data, k):
        """
        Convert Numpy 2D array to DataFrame with k PCA columns
        :param data: Numpy 2D array of data features
        :param k: Number of PCA components to label cols
        """
        reduced_columns = []
        for i in range(k):
            new_column = "PC" + str(i + 1)
            reduced_columns.append(new_column)
        df = pd.DataFrame(data=data, columns=reduced_columns)

        return df

    # Return original if # features <= # PCA components
    if len(list(X_train)) <= k:
        return X_train, X_test

    pca = PCA(n_components=k)
    reduced_training = pca.fit_transform(X_train)
    reduced_testing = pca.transform(X_test)
    print("\nPCA Analysis: Explained Variance Ratio")
    print(pca.explained_variance_ratio_)

    reduced_training = convert_to_df(reduced_training, k)
    reduced_testing = convert_to_df(reduced_testing, k)
    return reduced_training, reduced_testing
