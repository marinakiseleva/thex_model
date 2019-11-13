from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from models.base_model_mc.mc_base_model import MCBaseModel
from thex_data.data_clean import convert_str_to_list
from thex_data.data_consts import TARGET_LABEL, TREE_ROOT


class EnsembleModel(MCBaseModel, ABC):
    """
    Model that consists of K classifiers where is K is total number of all unique class labels. Each classifier compares a single class to remaining classes. Probabilities are reported unnormalized (just one versus all probabilities).
    """

    @abstractmethod
    def create_classifier(self, pos_class, X, y):
        """
        Initialize classifier, with positive class as positive class name
        :param pos_class: class_name that corresponds to TARGET_LABEL == 1
        :param X: DataFrame of features
        :param y: DataFrame with TARGET_LABEL column, 1 if it has class, 0 otherwise
        """
        pass

    def train_model(self):
        """
        Train K-models, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.class_labels = self.get_mc_unique_classes(self.y_train)

        # Create classifier for each class
        valid_classes = []
        for class_index, class_name in enumerate(self.class_labels):
            y_relabeled = self.get_class_data(class_name, self.y_train)
            positive_count = y_relabeled.loc[y_relabeled[TARGET_LABEL] == 1].shape[0]
            if positive_count < 3:
                print("WARNING: No model for " + class_name)
                continue

            print("\nClass Model: " + class_name)
            self.models[class_name] = self.create_classifier(
                class_name, self.X_train, y_relabeled)
            valid_classes.append(class_name)

        # Update class labels to only have classes for which we built models
        if len(valid_classes) != len(self.class_labels):
            print("\nWARNING: Not all class labels have classifiers.")
            self.class_labels = valid_classes

        return self.models

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def get_all_class_probabilities(self, normalized='independent'):
        """
        Get class probabilities for all test data.
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        all_probs = np.empty((0, len(self.class_labels)))
        for index, row in self.X_test.iterrows():
            row_p = self.get_class_probabilities(row)
            all_probs = np.append(all_probs, [list(row_p.values())], axis=0)

        return all_probs

    def get_class_probabilities(self, x, normalized='independent'):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :param normalized: Normalization technique; defaults to 'independent' which normalizes by summing over probabilities of ALL classes; ignores class hierarchy
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            class_prob = self.models[class_name].get_class_probability(x)
            probabilities[class_name] = class_prob
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = 0.001
                print("EnsembleModel get_class_probabilities NULL probability for " + class_name)

            if probabilities[class_name] < 0.0001:
                # Force min prob to 0.001 for future computation
                probabilities[class_name] = 0.001

        if normalized:
            probabilities = self.normalize_probabilities(probabilities, normalized)
        return probabilities

    def normalize_probabilities(self, probabilities, normalization_type):
        """
        Based on strategy
        :param probabilities: Dictionary from class names to likelihoods for single sample
        :param normalization_type: Type of normalization to apply.
            'unique' = each class is one versus all classifier, normalized altogether. disregards hierarchy
        """
        if normalization_type == 'independent':
            return self.norm_independent(probabilities)
            # 3. OPTIONAL: Cutoff probabilites below certain threshold
            # threshold = .9
            # for class_name in probabilities.keys():
            #     if probabilities[class_name] < threshold:
            #         probabilities[class_name] = 0
            # return probabilities
        elif normalization_type == 'conditional_siblings':
            # Normalize across disjoint sets of siblings
            probabilities = self.norm_siblings(probabilities)
            # Compute conditional probabilities based on hierarchy
            probabilities = self.norm_top_down(probabilities)
            return probabilities

        elif normalization_type == 'level_based':
            return self.norm_level(probabilities)

    def norm_level(self, probabilities):
        """
        Normalize across the class hierarchy levels. So, each is normalized by dividing by the sum of all probabilities at its level of the class hierarchy
        """

        levels = list(self.level_classes.keys())[1:]
        for level in levels:
            cur_level_classes = set(self.level_classes[level]).intersection(
                set(probabilities.keys()))
            level_total = 0
            for class_name in cur_level_classes:
                level_total += probabilities[class_name]

            # Normalize over level sum
            for class_name in cur_level_classes:
                probabilities[class_name] = probabilities[class_name] / level_total

        return probabilities

    def norm_independent(self, probabilities):
        """
        Normalize across probabilities, treating each as independent. So, each is normalized by dividing by the sum of all probabilities.
        :param probabilities: Dict from class names to probabilities, already normalized across disjoint sets
        """
        total = sum(probabilities.values())
        norm_probabilities = {class_name: prob /
                              total for class_name, prob in probabilities.items()}

        return norm_probabilities

    def norm_top_down(self, probabilities):
        """
        Compute the conditional probabilities of classes based on the hierarchy in a top-down approach. For each level of the hierarchy, compute probability of class as sibling-normalized binary classifier probability * parent probability.
        :param probabilities: Dict from class names to probabilities, already normalized across disjoint sets
        """

        # Compute conditional probabilities
        for current_level in range(max(self.class_levels.values())):
            for class_name, probability in probabilities.items():
                if self.class_levels[class_name] == current_level and class_name in probabilities:
                    probabilities[
                        class_name] *= self.get_parent_prob(class_name, probabilities)

        return probabilities

    def norm_bottom_up(self, probabilities):
        """
        Normalizes probabilities by using hierarchy to get probability of parents based on children, in bottom-up approach. Update inner node probabilities to reflect contraints of the class hierarchy. Leaf nodes are unchanged.
        :param probabilities: Dict from class names to probabilities, already normalized across disjoint sets
        """
        # Set inner-node probabilities
        levels = list(self.level_classes.keys())[1:]
        # Start at bottom of tree
        for level in reversed(levels):
            # Get all classes at this level & in model
            cur_level_classes = set(self.level_classes[level]).intersection(
                set(probabilities.keys()))
            Z = 0
            inner_classes = []
            for class_name in cur_level_classes:
                children = self.tree._get_children(class_name)
                children = set(children).intersection(set(probabilities.keys()))

                if len(children) > 0:  # Inner node
                    inner_classes.append(class_name)
                    # p(y\hat | y = 1)
                    pos_prob = self.models[class_name].pos_dist.pdf(
                        probabilities[class_name])

                    # p(y\hat | y = 0)
                    neg_prob = self.models[class_name].neg_dist.pdf(
                        probabilities[class_name])

                    # p(y) is probability of parent given childen, based on frequencies,
                    # basically the weighted sum
                    p_y = 0
                    for child in children:
                        # prob of parent i given child
                        p_i_G_child = self.parent_probs[(class_name, child)]
                        p_y += (p_i_G_child * probabilities[child])

                    p_y_hat_G_y = (pos_prob / (pos_prob + neg_prob))

                    hier_prob = p_y_hat_G_y * p_y

                    Z += hier_prob  # Normalizing constant

                    probabilities[class_name] = hier_prob

            for k in inner_classes:
                # Normalize across inner nodes at this level
                if Z > 0:
                    probabilities[k] = probabilities[k] / Z
                else:
                    probabilities[k] = 0

                if np.isnan(probabilities[k]):
                    print("NAN Value for " + k)

        return probabilities

    def norm_siblings(self, probabilities):
        """
        Normalize over set of all siblings
        :param probabilities: Dict from class names to probabilities
        """
        for primary_class in probabilities.keys():
            parent = self.tree._get_parent(primary_class)
            sibling_sum = 0
            num_siblings = 0
            for class_name in probabilities.keys():
                if self.tree._get_parent(class_name) == parent:
                    sibling_sum += probabilities[class_name]
                    num_siblings += 1

            # If there is only 1 class in set, do not normalize
            if num_siblings > 1:
                probabilities[primary_class] = probabilities[primary_class] / sibling_sum
        return probabilities

    def norm_level(self, probabilities):
        # Normalize over level
        for level in self.level_classes.keys():
            cur_level_classes = self.level_classes[level]
            # Normalize over this set of columns in probabilities
            level_sum = 0
            num_classes = 0
            for c in probabilities.keys():
                if c in cur_level_classes:
                    num_classes += 1
                    level_sum += probabilities[c]

            # Normalize by dividing each over sum
            for c in probabilities.keys():
                # If there is only 1 class in set, do not normalize
                if c in cur_level_classes and num_classes > 1:
                    probabilities[c] = probabilities[c] / level_sum

        return probabilities

    def get_parent_prob(self, class_name, probabilities):
        """
        Recurse up through tree, getting parent prob until we find a valid one. For example, there may only be CC, II, II P in CC so we need to inherit the probability of CC.
        """
        if class_name == TREE_ROOT:
            return 1
        elif self.tree._get_parent(class_name) in probabilities:
            return probabilities[self.tree._get_parent(class_name)]
        else:
            # Get next valid parent prob
            return self.get_parent_prob(self.tree._get_parent(class_name),
                                        probabilities)

    def get_mc_class_metrics(self):
        """
        Overriding get_mc_class_metrics in MCBaseModel to function for one versus all classification. Collects metrics based on maximum probabilities across hierarchy levels.
        Save TP, FN, FP, TN, and BS(Brier Score) for each class.
        Brier score: (1 / N) * sum(probability - actual) ^ 2
        Log loss: -1 / N * sum((actual * log(prob)) + (1 - actual)(log(1 - prob)))
        self.y_test has TARGET_LABEL column with string list of classes per sample
        """
        class_probs_np = self.get_all_class_probabilities()
        class_probabilities = pd.DataFrame(class_probs_np, columns=self.class_labels)
        class_accuracies = {}

        for current_level in range(1, max(self.level_classes.keys())):
            classes = self.level_classes[current_level]
            cur_level_classes = list(set(self.class_labels).intersection(classes))
            for class_name in cur_level_classes:
                TP = 0  # True Positives
                FN = 0  # False Negatives
                FP = 0  # False Positives
                TN = 0  # True Negatives
                BS = 0  # Brier Score
                LL = 0  # Log Loss
                cur_class_probs = class_probabilities[cur_level_classes]
                for index, row in cur_class_probs.iterrows():
                    actual_classes = self.y_test.iloc[index][TARGET_LABEL]
                    # Get class in cur_level_classes with maximum probability
                    max_class = row.idxmax()

                    if class_name in actual_classes:
                        # current class is in target, and maximum probability was given
                        # to it
                        if max_class == class_name:
                            TP += 1
                        else:
                            FN += 1
                    else:  # current class is not in target class
                        # Assigned max prob to current class, which is not in target
                        if max_class == class_name:
                            FP += 1
                        # Assigned max prob to another class
                        else:
                            TN += 1

                class_accuracies[class_name] = {"TP": TP,
                                                "FN": FN,
                                                "FP": FP,
                                                "TN": TN,
                                                "BS": BS,
                                                "LL": LL}

        return class_accuracies
