import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import StratifiedKFold


import utilities.utilities as thex_utils
from thex_data.data_consts import TARGET_LABEL, CPU_COUNT


class MultiKDEClassifier():
    """
    Multiclass KDE classifier
    """

    def __init__(self, X, y, class_labels):
        """
        Init classifier through training
        """
        self.name = "Multiclass Multivariate KDE"
        self.class_labels = class_labels
        self.clfs = self.train(X, y)
        # self.train_together(X, y)

    def get_class_data(self, class_name, y):
        """
        Return DataFrame like y except that TARGET_LABEL values have been replaced with 0 or 1. 1 if class_name is in list of labels.
        :param class_name: Positive class
        :return: y, relabeled
        """
        labels = []  # Relabeled y
        for df_index, row in y.iterrows():
            cur_classes = thex_utils.convert_str_to_list(row[TARGET_LABEL])
            label = 1 if class_name in cur_classes else 0
            labels.append(label)
        relabeled_y = pd.DataFrame(labels, columns=[TARGET_LABEL])
        return relabeled_y

    def get_one_hot(self, y):
        """
        Convert list of labels to one-hot encodings in order of self.class_labels; output 2D numpy array where each row is one-hot encoding for original label
        """
        num_labels = len(self.class_labels)
        onehot_rows = []
        for index, row in y.iterrows():
            onehot = [0 for _ in range(num_labels)]
            for index, class_label in enumerate(self.class_labels):
                if class_label in row[TARGET_LABEL]:
                    onehot[index] = 1
            onehot_rows.append(onehot)
        output = np.array(onehot_rows)
        return output

    def brier_multi(self, targets, probs):
        """
        Brier score loss for multiple classes:
        https://www.wikiwand.com/en/Brier_score#/Original_definition_by_Brier
        :param targets: 2D numpy array of one hot vectors
        :param probs: 2D numpy array of probabilities, same order of classes as targets
        """
        return np.mean(np.sum((probs - targets)**2, axis=1))

    def evaluate_fold(self, X, y):
        """
        Get Brier Score Loss for this set of KDEs
        """

        pos_probs = []
        for index, x in X.iterrows():
            probs_map = self.get_class_probabilities(x)
            pos_probs.append(list(probs_map.values()))

        pos_probs = np.array(pos_probs)
        onehot_y = self.get_one_hot(y)
        # Evaluate using Brier Score Loss
        loss = self.brier_multi(onehot_y, pos_probs)
        return loss

    def fit_folds(self, X, y, leaf_size, bandwidth, kernel):
        """
        Fit data using this bandwidth and kernel to every class, and report back brier score loss average over 3 folds
        """
        losses = []
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index].reset_index(
                drop=True), X.iloc[test_index].reset_index(drop=True)
            y_train, y_test = y.iloc[train_index].reset_index(
                drop=True), y.iloc[test_index].reset_index(drop=True)

            self.clfs = {}
            for class_name in self.class_labels:
                y_train_relabeled = self.get_class_data(class_name, y_train)
                X_pos = X_train.loc[y_train_relabeled[TARGET_LABEL] == 1]
                cur_kde = KernelDensity(bandwidth=bandwidth,
                                        leaf_size=leaf_size,
                                        metric='euclidean',
                                        kernel=kernel)
                cur_kde.fit(X_pos)
                self.clfs[class_name] = cur_kde

            # Evaluate using Brier Score Loss
            loss = self.evaluate_fold(X_test, y_test)
            losses.append(loss)
        # Average loss for this bandwidth across 3 folds
        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def train_together(self, X, y):
        """
        Fit kernel and bandwidth to all classes together, minimizing brier score loss instead of maximizing log likelihood of fit
        :param X: DataFrame of training data features
        :param y: DataFrame of TARGET_LABEL with 1 for pos_class and 0 for not pos_class
        """
        leaf_size = 40
        bandwidths = np.linspace(0.0001, 5, 100)
        kernels = ['exponential', 'gaussian', 'tophat',
                   'epanechnikov', 'cosine', 'linear']
        best_cv_loss = 1000
        best_cv_bw = None
        best_kernel = None
        for kernel in kernels:
            for bandwidth in bandwidths:
                loss = self.fit_folds(X, y, leaf_size, bandwidth, kernel)

                # Reset best BW overall.
                if loss < best_cv_loss:
                    best_cv_loss = loss
                    best_cv_bw = bandwidth
                    best_kernel = kernel

        # Define models based on best bandwidth
        print("Best bandwidth " + str(best_cv_bw))
        print("Best kernel " + str(best_kernel))
        print("With score: " + str(best_cv_loss))

        # Define each class KDE using optimal hyperparameters
        self.clfs = {}
        for class_name in self.class_labels:
            y_relabeled = self.get_class_data(class_name, y)
            X_pos = X.loc[y_relabeled[TARGET_LABEL] == 1]
            cur_kde = KernelDensity(bandwidth=best_cv_bw,
                                    leaf_size=leaf_size,
                                    metric='euclidean',
                                    kernel=best_kernel)
            cur_kde.fit(X_pos)
            self.clfs[class_name] = cur_kde

    def train(self, X, y):
        mc_kdes = {}
        for class_name in self.class_labels:
            print("\n\nTraining: " + class_name)
            y_relabeled = self.get_class_data(class_name, y)
            X_pos = X.loc[y_relabeled[TARGET_LABEL] == 1]
            mc_kdes[class_name] = self.fit_class(X_pos)

        return mc_kdes

    def fit_class(self, X):
        """
        Fit KDE to X & select bandwidth by maximizing log-likelihood 
        :return: best fitting KDE
        """
        # Create grid to get optimal bandwidth
        grid = {'bandwidth': np.linspace(0, 1, 100)}
        num_cross_folds = 3  # number of folds in a (Stratified)KFold
        kde = KernelDensity(leaf_size=10,
                            metric='euclidean',
                            kernel='exponential')
        clf_optimize = GridSearchCV(estimator=kde,
                                    param_grid=grid,
                                    cv=num_cross_folds,
                                    iid=True,
                                    n_jobs=CPU_COUNT
                                    )
        clf_optimize.fit(X)
        clf = clf_optimize.best_estimator_

        print("Optimal KDE Parameters: " + str(clf_optimize.best_params_) +
              " \nwith log probability density (log-likelihood): " + str(clf.score(X)))

        return clf

    def get_class_probabilities(self, x):
        """
        Get probability of each class for this sample x. Probability of class i  = density_i / (sum_i^N density_i). 
        :param x: Pandas Series (row of DF) of features
        """
        density_sum = 0
        probabilities = {}
        for class_name in self.class_labels:
            class_density = np.exp(self.clfs[class_name].score_samples([x.values]))[0]
            probabilities[class_name] = class_density
            density_sum += class_density

        # Normalize
        probabilities = {k: probabilities[k] / density_sum for k in probabilities.keys()}

        MIN_PROB = 0.0001  # Min probability to avoid overflow
        for class_name in self.class_labels:
            if np.isnan(probabilities[class_name]):
                probabilities[class_name] = MIN_PROB
                # print(self.name + " NULL probability for " + class_name)

            if probabilities[class_name] < MIN_PROB:
                probabilities[class_name] = MIN_PROB

        return probabilities
