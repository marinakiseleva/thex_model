from models.base_model.base_model import BaseModel
from thex_data.data_consts import class_to_subclass as hierarchy
from thex_data.data_consts import TARGET_LABEL, code_cat, cat_code
from model_performance.performance import *

from hmc import hmc
from hmc import metrics
from hmc.datasets import *


class HMCModel(BaseModel):
    """
    Hierarchical Multi-Label Classifier using decision trees, based on the HMC in Vens, et al 2008. 
    """

    def __init__(self, cols=None, col_match=None, test_on_train=False, incl_redshift=False, folds=3):
        self.name = "HMC Decisioning Tree Model"
        self.run_model(cols, col_match, test_on_train, incl_redshift, folds)

    def train_model(self):
        """
        Builds hierarchy and tree, and fits training data to it.
        """
        y_test_strs = self.convert_target(self.y_test)
        y_train_strs = self.convert_target(self.y_train)
        self.hmc_hierarchy = self.init_hierarchy()
        self.tree = hmc.DecisionTreeHierarchicalClassifier(self.hmc_hierarchy)
        tree = self.tree.fit(self.X_train, y_train_strs)

    def test_model(self):
        predicted_classes = self.tree.predict(self.X_test)
        # predicted_classes are in terms of class names, so need to be converted back to codes
        # evaluate_tree(predictions y_test_strs)
        prediction_codes = [cat_code[pred_class] for pred_class in predicted_classes]
        return prediction_codes

    def convert_target(self, class_codes):
        """
        Converts transient codes to their category names for the tree (because the hierarchy is defined in terms of names, not codes)
        :param class_codes: DataFrame of class_codes, with TARGET_LABEL as column
        """
        class_names = class_codes.copy()
        class_names[TARGET_LABEL] = class_codes[
            TARGET_LABEL].apply(lambda x: code_cat[x])
        return class_names

    def init_hierarchy(self):
        """
        Initialize hierarchy for HMC.
        """
        # TTypes  = top-level class in data_consts map.
        hmc_hierarchy = hmc.ClassHierarchy("TTypes")
        for parent in hierarchy.keys():
            # hierarchy maps parents to children, so get all children
            list_children = hierarchy[parent]
            for child in list_children:
                # Nodes are added with child parent pairs
                try:
                    hmc_hierarchy.add_node(child, parent)
                except ValueError as e:
                    print(e)
        hmc_hierarchy.print_()
        return hmc_hierarchy

    def evaluate_tree(self, predictions, y_test_names):
        """
        Evaluates tree performance using Tree metrics
        :param y_test_names: y_test labels in class name form
        """
        dth_accuracy = self.tree.score(self.X_test, y_test_names)

        print("Accuracy: %s" % metrics.accuracy_score(
            self.hmc_hierarchy, y_test_names, predictions))
        print("-------------------------- Ancestors -------------------------- ")
        print("Precision Ancestors: %s" %
              metrics.precision_score_ancestors(self.hmc_hierarchy, y_test_names, predictions))
        print("Recall Ancestors: %s" % metrics.recall_score_ancestors(
            self.hmc_hierarchy, y_test_names, predictions))
        print("F1 Ancestors: %s" % metrics.f1_score_ancestors(
            self.hmc_hierarchy, y_test_names, predictions))
        print("-------------------------- Descendants -------------------------- ")
        print("Precision Descendants: %s" %
              metrics.precision_score_descendants(self.hmc_hierarchy, y_test_names, predictions))
        print("Recall Descendants: %s" % metrics.recall_score_descendants(
            self.hmc_hierarchy, y_test_names, predictions))
        print("F1 Descendants: %s" % metrics.f1_score_descendants(
            self.hmc_hierarchy, y_test_names, predictions))

        class_precisions = metrics.precision_score_hierarchy(
            self.hmc_hierarchy, y_test_names, predictions, level=1)
        class_accuracies = {}
        for tclass in class_precisions.keys():
            correct = class_precisions[tclass][0]
            total = class_precisions[tclass][1]
            if total != 0:
                class_accuracies[cat_code[tclass]] = correct / total
        plot_class_accuracy(class_accuracies, "Top-Level Hierarchy Performance")

        # print("Precision at the top level of the hierarchy: " + str(acc))

        prediction_codes = [cat_code[cc] for cc in predictions]
        actual_codes = [cat_code[cc] for cc in y_test_names[TARGET_LABEL].values]
        plot_confusion_matrix(actual_codes, prediction_codes,
                              normalize=True)

        compute_plot_class_accuracy(
            prediction_codes, actual_codes, "HMC Tree Accuracy")
