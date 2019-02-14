from sklearn.model_selection import KFold

from model_performance.performance import *
from models.base_model.cmd_interpreter import get_model_data

from models.tree_model.hmc_tree import train_tree, convert_target
from models.nb_model.nb_train import train_nb
from models.nb_model.nb_test import test_nb
from thex_data.data_prep import get_train_test, get_source_target_data
from thex_data.data_consts import cat_code, TARGET_LABEL


def run_models(X_train, y_train, X_test, y_test):
    """
    Run Naive Bayes and HMC Tree
    Return predictions of each on testing data
    """
    # Run Naive Bayes
    summaries, priors = train_nb(X_train, y_train)
    nb_predictions = test_nb(X_test, summaries, priors)

    # Run HMC Tree
    # Tree uses category names, not codes
    y_train_strs, y_test_strs = convert_target(y_train, y_test)
    tree, hmc_hierarchy = train_tree(X_train, y_train_strs)
    tree_predictions = [cat_code[cc] for cc in tree.predict(X_test)]

    return nb_predictions, tree_predictions


def aggregate_accuracies(fold_accuracy, k, y):
    """
    Aggregate accuracies from cross-validation
    """
    accuracy_per_class = {c: 0 for c in list(y[TARGET_LABEL].unique())}
    for class_accuracies in fold_accuracy:
        # class_accuracies maps class to accuracy as %
        for tclass in class_accuracies.keys():
            accuracy_per_class[tclass] += class_accuracies[tclass]
    # Divide each % by number of folds to get average accuracy
    return {c: acc / k for c, acc in accuracy_per_class.items()}


def run_cv(data_columns, incl_redshift=False, test_on_train=False, k=3):
    """
    Split data into k-folds and perform k-fold cross validation
    """
    X, y = get_source_target_data(data_columns, incl_redshift)
    kf = KFold(n_splits=k, shuffle=True, random_state=10)
    nb_recall, tree_recall = [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        data_type = ' Testing '
        if test_on_train:  # Update fields to test models on training data
            X_test = X_train.copy()
            y_test = y_train.copy()
            data_type = ' Training '

        nb_predictions, tree_predictions = run_models(X_train, y_train, X_test, y_test)

        # Save Naive Bayes accuracy
        nb_class_accuracies = get_class_accuracies(combine_dfs(nb_predictions, y_test))
        nb_recall.append(nb_class_accuracies)

        # Save tree accuracy
        tree_class_accuracies = get_class_accuracies(
            combine_dfs(tree_predictions, y_test))
        tree_recall.append(tree_class_accuracies)

    # Get average accuracy per class (mapping)
    avg_tree_acc = aggregate_accuracies(tree_recall, k, y)
    avg_nb_acc = aggregate_accuracies(nb_recall, k, y)
    plot_class_accuracy(
        avg_tree_acc, plot_title="HSC Tree: " + str(k) + "-fold CV on" + data_type + " data")
    plot_class_accuracy(
        avg_nb_acc, plot_title="Naive Bayes: " + str(k) + "-fold CV on" + data_type + " data")


def main(cv=True):
    """
    Runs comparison between Naive Bayes and Tree - using same data
    :param cv: Boolean for running 3-fold cross-validation. If false, it will run only once. 
    """
    col_list, incl_redshift, test_on_train = collect_args()
    if cv:
        run_cv(col_list, incl_redshift, test_on_train=test_on_train, k=3)
        return 1

    X_train, X_test, y_train, y_test = get_train_test(
        col_list, incl_redshift, split=0.4)

    data_type = ' Training ' if test_on_train else ' Testing '
    if test_on_train:
        X_test = X_train.copy()
        y_test = y_train.copy()

    nb_predictions, tree_predictions = run_models(X_train, y_train, X_test, y_test)

    compute_plot_class_accuracy(
        nb_predictions, y_test, plot_title="Naive Bayes Accuracy, on" + data_type + "Data")
    compute_plot_class_accuracy(tree_predictions, y_test,
                                plot_title="HMC Tree Accuracy, on" + data_type + "Data")

if __name__ == '__main__':
    main()
