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
    Hierarchical Multi-Label Classifier based on predictive clustering tree (PCT) and using bagging. Implementation of CLUS-HMC-ENS outlined in Kocev, Dzeroski 2010. Splits using reduction in class-weighted variance.
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "CLUS-HMC-ENS"
        data_args['transform_labels'] = False
        data_args['hierarchical_model'] = True
        self.cols = cols
        self.col_matches = col_matches
        self.user_data_filters = data_args
        # Save class labels to always access in same order
        self.class_labels = list(cat_code.keys())

    def train_model(self):
        """
        Builds hierarchy and tree, and fits training data to it.
        """
        labeled_samples, feature_value_pairs = self.setup_data()
        self.root = self.train(labeled_samples, feature_value_pairs, remaining_depth=300)

    def convert_class_vectors(self, df):
        """
        Convert labels of TARGET_LABEL column in passed-in DataFrame to class vectors
        """
        # Convert labels to class vectors, with 1 meaning it has that class, and 0
        # does not
        rows_list = []
        for df_index, row in df.iterrows():
            class_vector = [0] * len(self.class_labels)
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            for class_index, c in enumerate(self.class_labels):
                if c in cur_classes:
                    class_vector[class_index] = 1
            rows_list.append([class_vector])
        class_vectors = pd.DataFrame(rows_list, columns=[TARGET_LABEL])
        return class_vectors

    def get_sample_weights(self, labeled_samples):
        """
        Get weight of each sample (1/# of samples in class) and save in list with same order as labeled_samples
        :param labeled_samples: DataFrame of fetaures and TARGET_LABEL
        """
        label_counts = Counter(tuple(e) for e in labeled_samples[TARGET_LABEL].values)
        sample_weights = []
        for df_index, row in labeled_samples.iterrows():
            class_vector = tuple(row[TARGET_LABEL])
            class_count = label_counts[class_vector]
            sample_weights.append(1 / class_count)
        return sample_weights

    def setup_data(self):
        """
        Setup data parameters for model. Return
        labeled_samples : features and TARGET_LABEL, which for each row has the
        class_vector
        feature_value_pairs : list of [f,v] pairs
        """
        class_vectors = self.convert_class_vectors(self.y_train)
        # Add labels to training data for complete dataset
        labeled_samples = pd.concat([self.X_train, class_vectors], axis=1)

        # Initialize weight vectors
        tree = init_tree()
        class_level = assign_levels(tree, {}, tree.root, 1)
        self.class_weights = [1 / class_level[c] for c in self.class_labels]

        # Compute sample weights : Weigh each sample so that weights of samples in
        # each class sum to 1
        self.sample_weights = self.get_sample_weights(labeled_samples)

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
        return labeled_samples, feature_value_pairs

    def train(self, labeled_samples, remaining_feature_value_pairs, remaining_depth):
        """
        Constructs CLUS-HMC-ENS decision tree. Returns node at root of tree.
        :param labeled_samples: DataFrame with both features and label
        :param remaining_feature_value_pairs: list of feature/value pairs
        :param remaining_depth: # of levels tree is allowed to construct
        """
        num_samples = labeled_samples.shape[0]
        leaf_min = 5  # Minimum number of samples in a leaf
        if remaining_depth == 0 or self.is_unambiguous(labeled_samples) or len(remaining_feature_value_pairs) == 0 or num_samples <= leaf_min:
            return LeafNode(self.majority_class(labeled_samples))

        # get majority votes over remaining feature/value pairs
        fv_variance = {}
        # current_variance = self.get_variance(labeled_samples, None, None, None)
        for pair in remaining_feature_value_pairs:
            feature = pair[0]
            value = pair[1]
            # Get class variance of samples with (& without) this feature/value
            var = self.get_variance(labeled_samples, feature, value)
            fv_variance[tuple(pair)] = var

        # Split based on minimum variance
        best_feature_val = min(fv_variance, key=fv_variance.get)
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

    def get_set_variance(self, labeled_samples):
        """
        Get variance of dataset based on classes. Based on paper, variance is sum of squared distances between each sample and mean vector. Returns 10 if set is empty in order to penalize tree splits where one side has no data.
        :param labeled_samples: DataFrame of all features and TARGET_LABEL
        """
        if labeled_samples.shape[0] == 0:
            return 10
        class_data = labeled_samples[[TARGET_LABEL]]
        total_variance = 0
        mean_vector = self.get_mean_vector(class_data)
        for df_index, class_vector in class_data.iterrows():
            # Removed sample_weight logic since it over-emphasizes small classes.
            # sample_weight = self.sample_weights[df_index]
            d = self.get_weighted_distance(class_vector[TARGET_LABEL], mean_vector) ** 2
            total_variance += d  # * sample_weight
        num_samples = labeled_samples.shape[0]
        return total_variance / num_samples

    def get_variance(self, labeled_samples, feature, value):
        """
        Calculates variance of samples with this feature/value/label and samples without, and returns the sum of both. For each calculate:
        Var(S) = sum_k (d(v_k, v)^2 / |S|)  where d is distance
        :param labeled_samples: DataFrame of all features and TARGET_LABEL
        """
        # If there are no samples give default variance of 10 to discourage an
        # empty split
        if labeled_samples.shape[0] == 0:
            return 10

        if feature is None and value is None:
            # Calculate variance among all labeled_samples
            return self.get_set_variance(labeled_samples)

        # Filter samples to those with feature/value
        samples_greater, samples_less = self.split_samples(
            labeled_samples.copy(), feature, value)

        variance_samples_with = self.get_set_variance(samples_greater)
        variance_samples_without = self.get_set_variance(samples_less)

        return variance_samples_with + variance_samples_without

    def is_unambiguous(self, labeled_samples):
        """
        Return True if all samples have the same label; otherwise False
        :param labeled_samples: DataFrame of all features and TARGET_LABEL
        """
        label_counts = Counter(tuple(e) for e in labeled_samples[TARGET_LABEL].values)
        if len(list(label_counts.keys())) == 1:
            return True
        return False

    def split_samples(self, labeled_samples, feature, value):
        """
        Split samples into those with >= value (samples_greater) and samples < value (samples_less).
        """
        samples_greater = labeled_samples[labeled_samples[feature] >= value]
        samples_less = labeled_samples[labeled_samples[feature] < value]
        return samples_greater, samples_less

    def get_mean_vector(self, class_df):
        """
        Get average class vector from samples
        :param class_df: DataFrame of TARGET_LABEL column only
        """

        # Convert Series to array of class vector (each row is a class vector)
        class_vectors_array = np.array(class_df.values.tolist())
        # Get average row (class vector)
        avg_vector = np.mean(class_vectors_array, axis=0)
        return avg_vector[0]

    def get_weighted_distance(self, v, mean_v):
        """
        Get weighted Euclidean distance of data between two vectors, using weight vector w
        d(v_1, v_2) = sqrt (sum_i w(c_i) dot (v_1,i - v_2,i)^2)   where v_k,i is ith component of class vector v_k and  w(c_i) is vector of weights for each class
        :param v: class vector
        :param mean_v: mean class vector in set we are comparing against
        """
        if self.class_weights is None:
            raise ValueError(
                "Need to set up mapping between classes and weights in CLUS-HMC-ENS")
            exit(-1)
        dist = (np.dot(self.class_weights, (np.square(v - mean_v))))**(1 / 2)
        return dist

    def majority_class(self, labeled_samples):
        """
        Paper says to use mean vector as majority class leaf assignment; each element in output vector is proportion of examples belonging to that class
        # v = [0.5, 0.9, 0.1] for example
        :param labeled_samples: DataFrame of all features and TARGET_LABEL
        """
        return self.get_mean_vector(labeled_samples[[TARGET_LABEL]])

    def test_model(self):
        """
        Test model on self.y_train data.
        """
        predictions = []
        for index, test_point in self.X_test.iterrows():
            predicted_class_vector = self.test_sample(self.root, test_point)
            predictions.append([predicted_class_vector])

        self.predictions = pd.DataFrame(predictions, columns=['predicted_class'])
        self.evaluate_model()

        return self.predictions

    def test_sample(self, node, testing_features):
        if type(node) == LeafNode:
            return node.guess

        # Select child node based on feature value in test_point
        test_value = testing_features[node.feature]
        if node.feature_value >= test_value:
            # Continue down to samples with this feature/value pair
            return self.test_sample(node.sample_greater, testing_features)
        else:
            return self.test_sample(node.sample_less, testing_features)

    def evaluate_model(self):
        # Convert actual labels to class vectors for comparison
        self.y_test_vectors = self.convert_class_vectors(self.y_test)
        self.get_recall_scores()

    def get_recall_scores(self):
        """
        Get recall of each class
        """
        class_recalls = {label: 0 for label in self.class_labels}
        for class_index, class_name in enumerate(self.class_labels):
            class_recalls[class_name] = self.get_class_recall(class_index)
        print("\n\nRecall Performance\n")
        for key in class_recalls.keys():
            print(str(key) + " : " + str(round(class_recalls[key], 2)))

    def get_class_recall(self, class_index):
        """
        Get recall of class passed in using actual and predicted class vectors. Recall is TP/TP+FN
        """
        threshold = 0.4
        row_count = self.y_test_vectors.shape[0]
        TP = 0  # True positive count, predicted = actual = 1
        FN = 0  # False negative count, predicted = 0, actual = 1
        for index in range(row_count - 1):

            predicted_classes = self.predictions.iloc[index]['predicted_class']
            actual_classes = self.y_test_vectors.iloc[index][TARGET_LABEL]

            actual_class = actual_classes[class_index]
            predicted_class = predicted_classes[class_index]

            if actual_class >= threshold:
                if actual_class == predicted_class:
                    TP += 1
                elif actual_class != predicted_class:
                    FN += 1
        denominator = TP + FN
        class_recall = TP / (TP + FN) if denominator > 0 else 0
        return class_recall

    def get_class_probabilities(self, x):
        print("\n\n need to implement get_class_probabilities for CLUS-HMC-ENS \n")
        sys.exit(-1)
