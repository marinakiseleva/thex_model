from collections import Counter
import numpy as np
import pandas as pd
import sys

from thex_data.data_clean import init_tree, assign_levels, convert_str_to_list
from thex_data.data_consts import class_to_subclass as hierarchy
from thex_data.data_consts import TARGET_LABEL, cat_code

from models.base_model.base_model import BaseModel
from models.clus_hmc_ens_model.nodes import *


class CLUSHMCENS(BaseModel):
    """
    Hierarchical Multi-Label Classifier based on predictive clustering tree (PCT) and using bagging. Implementation of CLUS-HMC-ENS outlined in Kocev, Dzeroski 2010.
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "CLUS-HMC-ENS"
        data_args['transform_labels'] = False
        self.cols = cols
        self.col_matches = col_matches
        self.user_data_filters = data_args

    def train_model(self):
        """
        Builds hierarchy and tree, and fits training data to it.
        """
        # Convert labels to class vectors, with 1 meaning it has that class
        self.class_labels = list(cat_code.keys())
        rows_list = []
        for df_index, row in self.y_train.iterrows():
            class_vector = [0] * len(self.class_labels)
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            for class_index, c in enumerate(self.class_labels):
                if c in cur_classes:
                    class_vector[class_index] = 1
            rows_list.append([class_vector])
        class_vectors = pd.DataFrame(rows_list, columns=[TARGET_LABEL])

        # Add labels to training data for complete dataset
        labeled_samples = pd.concat([self.X_train, class_vectors], axis=1)

        # Initialize weight vector
        tree = init_tree()
        class_level = assign_levels(tree, {}, tree.root, 1)
        self.class_weights = [class_level[c] for c in self.class_labels]

        # For each feature, find up to 10 random values to use as feature/value pairs
        feature_value_pairs = []
        unique_features = list(self.X_train)
        num_values = 10
        for f in unique_features:
            unique_values = list(labeled_samples[f].unique())
            if len(unique_values) > num_values:
                min_val = min(unique_values)
                max_val = max(unique_values)

                unique_values = np.linspace(min_val, max_val, num_values)
            # Add each feature/value pair to dict
            for v in unique_values:
                feature_value_pairs.append([f, v])

        # labeled_samples: features and TARGET_LABEL, which for each row has the
        # class_vector
        # feature_value_pairs: list of [f,v] pairs
        root = self.train(
            labeled_samples, feature_value_pairs, remaining_depth=300)

    def train(self, labeled_samples, remaining_feature_value_pairs, remaining_depth):
        """
        Constructs CLUS-HMC-ENS decision tree. Returns node at root of tree.
        :param labeled_samples: DataFrame with both features and label
        :param remaining_feature_value_pairs: list of feature/value pairs
        :param remaining_depth: # of levels tree is allowed to construct
        """
        if remaining_depth == 0 or self.is_unambiguous(labeled_samples) or len(remaining_feature_value_pairs) == 0:
            return LeafNode(self.majority_class(labeled_samples))

        # get majority votes over remaining feature/value pairs
        set_variance = {}
        current_variance = self.get_variance(labeled_samples, None, None, None)
        labels = self.get_labels(labeled_samples)
        for pair in remaining_feature_value_pairs:
            feature = pair[0]
            value = pair[1]
            scores = []  # List of scores for each label
            for label_set in labels:
                # Get variance of samples with (& without) this feature/value/label
                scores.append(self.get_variance(
                    labeled_samples, feature, value, label_set))
            # Save minimum-most variance among all classes
            set_variance[tuple(pair)] = min(scores)

        # Feature/value that reclusters data with minimum-most variance
        best_feature_val = min(set_variance, key=set_variance.get)
        remaining_feature_value_pairs.remove(list(best_feature_val))
        # Split samples into those with feature/value and those without
        samples_greater, samples_less = self.split_samples(
            labeled_samples, best_feature_val[0], best_feature_val[1])

        # If split resulted in 0 samples -> return leaf node with best current guess

        if len(samples_greater) == 0:
            sample_greater = LeafNode(self.majority_class(labeled_samples))
        else:
            sample_greater = self.train(
                samples_greater, remaining_feature_value_pairs, remaining_depth - 1)
        if len(samples_less) == 0:
            sample_less = LeafNode(self.majority_class(labeled_samples))
        else:
            sample_less = self.train(
                samples_less, remaining_feature_value_pairs, remaining_depth - 1)

        return InternalNode(best_feature_val,  sample_greater, sample_less)

    def get_labels(self, labeled_samples):
        """
        Returns list of labels to iterate over in training
        """
        label_counts = Counter(tuple(e) for e in labeled_samples[TARGET_LABEL].values)
        unique_labels = []
        for l in label_counts.keys():
            unique_labels.append(list(l))
        return unique_labels

    def get_class_data(self, labeled_samples):
        """
        Drops all feature information, returning just class information for data
        """
        return labeled_samples[[TARGET_LABEL]]

    def get_variance(self, labeled_samples, feature, value, label_set):
        """
        Calculates variance of samples with this feature/value/label and samples without, and returns the sum of both. For each calculate:
        Var(S) = sum_k (d(v_k, v)^2 / |S|)  where d is distance 

        """
        # If there are no samples give default variance of 10 to discourage an
        # empty split
        if labeled_samples.shape[0] == 0:
            return 10

        def get_set_variance(dataset):
            """
            Get variance of dataset based on classes 
            """
            if dataset.shape[0] == 0:
                return 10
            class_data = self.get_class_data(dataset)
            total_variance = 0
            mean_vector = self.get_mean_vector(class_data)
            for df_index, class_vector in class_data.iterrows():
                total_variance += self.get_weighted_distance(
                    class_vector[TARGET_LABEL], mean_vector) ** 2
            S = 0
            for i in mean_vector:
                S += i**2

            return total_variance / (S**(1 / 2))

        if feature is None and value is None and label_set is None:
            # Calculate variance among all labeled_samples
            return get_set_variance(labeled_samples)

        # Filter samples to those with feature/value/label_set
        samples_greater, samples_less = self.split_samples(
            labeled_samples.copy(), feature, value, label_set)

        variance_samples_with = get_set_variance(samples_greater)
        variance_samples_without = get_set_variance(samples_less)

        return variance_samples_with + variance_samples_without

    def is_unambiguous(self, labeled_samples):
        """
        Return True if all samples have the same label; otherwise False
        """
        label_counts = Counter(tuple(e) for e in labeled_samples[TARGET_LABEL].values)
        if len(list(label_counts.keys())) == 1:
            return True
        return False

        # class_data = self.get_class_data(labeled_samples)
        # # If max and min are same in each column, there is only 1 unique row in
        # # the whole set
        # max_vals = pd.DataFrame(class_data.values.max(
        #     0)[None, :], columns=class_data.columns)
        # min_vals = pd.DataFrame(class_data.values.min(
        #     0)[None, :], columns=class_data.columns)

        # return max_vals.equals(min_vals)

    def split_samples(self, labeled_samples, feature, value, label_set=None):
        """
        Split samples into those with >= value (samples_greater) and samples < value (samples_less). 
        """
        if label_set is not None:
            # Filter down to just this label
            labeled_samples['targ_label'] = labeled_samples[TARGET_LABEL].apply(
                lambda x: 1 if x == label_set else 0)
            samples_greater = labeled_samples[(labeled_samples[
                feature] >= value) & (labeled_samples['targ_label'] == 1)]
            samples_less = labeled_samples[(labeled_samples[
                feature] < value) & (labeled_samples['targ_label'] == 1)]
            return samples_greater, samples_less

        samples_greater = labeled_samples[labeled_samples[feature] >= value]
        samples_less = labeled_samples[labeled_samples[feature] < value]
        return samples_greater, samples_less

    def get_mean_vector(self, class_vectors):
        """
        Get average class vector from samples
        """

        # Convert Series to array of class vector (each row is a class vector)
        class_vectors_array = np.array(class_vectors.values.tolist())
        # Get average row (class vector)
        avg_vector = np.mean(class_vectors_array, axis=0)
        return avg_vector[0]

    def get_weighted_distance(self, v1, v2):
        """ 
        Get weighted Euclidean distance of data between two vectors, using weight vector w
        d(v_1, v_2) = sqrt (sum_i w(c_i) dot (v_1,i - v_2,i)^2)   where v_k,i is ith component of class vector v_k and  w(c_i) is vector of weights for each class
        :param v1: class vector 1
        :param v2: class vector 2
        """
        if self.class_weights is None:
            raise ValueError(
                "Need to set up mapping between classes and weights in CLUS-HMC-ENS")
            exit(-1)
        return (np.dot(self.class_weights, (np.square(v1 - v2))))**(1 / 2)

    def majority_class(self, labeled_samples):
        """
        Returns most frequent labels (as class vector)
        """
        label_counts = Counter(tuple(e) for e in labeled_samples[TARGET_LABEL].values)

        majority_class_vector = list(max(label_counts, key=label_counts.get))
        return majority_class_vector
        # # Split class vector into columns
        # class_data = pd.DataFrame(
        #     labeled_samples[TARGET_LABEL].values.tolist(), columns=self.class_labels)
        # print(list(class_data))

        # # Get most frequent row
        # class_counts = (class_data.groupby(class_data.columns.tolist()
        #                                    ).size().sort_values(ascending=False)).head(1)
        # majority_class_df = pd.DataFrame(class_counts).reset_index()

        # # Convert row back to list
        # majority_class_vector = majority_class_df[self.class_labels].values.tolist()[0]
        # print(majority_class_vector)

    def test_model(self):
        print("\n\n need to implement test_model for CLUS-HMC-ENS \n")
        sys.exit(-1)

    def get_class_probabilities(self, x):
        print("\n\n need to implement get_class_probabilities for CLUS-HMC-ENS \n")
        sys.exit(-1)
