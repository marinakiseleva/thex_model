"""
Independent Model

Ensemble of binary classifiers with resulting probabilities normalized across all classes. Assumes all classes are independent and are not related by any class hierarchy.

"""

import numpy as np
from mainmodel.mainmodel import MainModel
from classifiers.binary.optbinary import OptimalBinaryClassifier


class OvAModel(MainModel):

    def __init__(self, **data_args):
        """
        Initialize Independent Model - call init method of parent class
        """
        self.name = "OVA Classifier"
        # Use output of Binary Classifier model for OvAModel - just need to
        # normalize those probabilities.
        if 'init_from_binary' in data_args.keys():
            self.init_model = data_args['init_from_binary']
        else:
            self.init_model = None
        super(OvAModel, self).__init__(**data_args)

    def run_cfv(self, start_time):
        """
        Overwrite MainModel function to see if results from BinaryModel were passed in
        """
        if self.init_model is not None:
            # Normalize probabilities per row (Except last column which is label)
            print("Using Binary Model results.")
            self.class_labels = self.init_model.class_labels
            self.class_counts = self.init_model.class_counts
            self.num_folds = self.init_model.num_folds
            self.X = self.init_model.X
            self.y = self.init_model.y
            self.results = self.init_model.results

            a = np.copy(self.init_model.results)
            new_results = []
            for index in range(self.init_model.num_folds):
                trial_results = []
                for row_index, row in enumerate(a[index]):
                    num_classes = len(self.init_model.class_labels)

                    prob_sum = 0  # compute normalizing sum
                    for class_index in range(num_classes):
                        prob_sum += row[class_index]

                    new_probs = []
                    for class_index in range(num_classes):
                        new_probs.append(row[class_index] / prob_sum)
                    new_probs.append(row[num_classes])  # Add label
                    trial_results.append(new_probs)
                new_results.append(np.array(trial_results, dtype='object'))
            return new_results
        else:
            return super(OvAModel, self).run_cfv(start_time)

    def run_trials(self, X, y, num_runs):
        """
        Overwrite MainModel function to see if results from BinaryModel were passed in
        """
        if self.init_model is not None:
            # Normalize probabilities per row (Except last column which is label)
            print("Using Binary Model results.")
            self.class_labels = self.init_model.class_labels
            self.class_counts = self.init_model.class_counts
            self.num_runs = self.init_model.num_runs
            self.X = self.init_model.X
            self.y = self.init_model.y

            a = np.copy(self.init_model.results)
            new_results = []
            for index in range(self.init_model.num_runs):
                trial_results = []
                for row_index, row in enumerate(a[index]):
                    num_classes = len(self.init_model.class_labels)

                    prob_sum = 0  # compute normalizing sum
                    for class_index in range(num_classes):
                        prob_sum += row[class_index]

                    new_probs = []
                    for class_index in range(num_classes):
                        new_probs.append(row[class_index] / prob_sum)
                    new_probs.append(row[num_classes])  # Add label
                    trial_results.append(new_probs)
                new_results.append(np.array(trial_results, dtype='object'))
            return new_results
        else:
            return super(OvAModel, self).run_trials(X, y, num_runs)

    def train_model(self, X_train, y_train):
        """
        Train model using training data. All classes are a single disjoint set, so create a binary classifier for each one
        """

        self.models = {}
        for class_index, class_name in enumerate(self.class_labels):
            y_relabeled = self.relabel_class_data(class_name, y_train)
            print("\nClass Model: " + class_name)
            self.models[class_name] = OptimalBinaryClassifier(
                class_name, X_train, y_relabeled, self.nb, self.dir)

    def get_class_probabilities(self, x):
        """
        Calculates probability of each transient class for the single test data point (x).
        :param x: Pandas DF row of features
        :return: map from class_name to probabilities
        """
        probabilities = {}
        for class_index, class_name in enumerate(self.class_labels):
            probabilities[class_name] = self.models[class_name].get_class_probability(x)
            MIN_PROB = 0.0001  # Min probability to avoid overflow
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
                print(self.name + " NULL probability for " + class_name)

            if probabilities[class_name] < MIN_PROB:
                probabilities[class_name] = MIN_PROB

        # Normalize
        if self.normalize:
            probabilities = self.normalize_probs(probabilities)

        return probabilities

    def normalize_probs(self, probabilities):
        """
        Normalize across probabilities, treating each as independent. So, each is normalized by dividing by the sum of all probabilities.
        :param probabilities: Dict from class names to probabilities, already normalized across disjoint sets
        """
        total = sum(probabilities.values())
        norm_probabilities = {class_name: prob /
                              total for class_name, prob in probabilities.items()}

        return norm_probabilities
