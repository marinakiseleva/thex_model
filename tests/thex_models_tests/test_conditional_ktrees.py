import sys
import unittest
import pandas as pd
sys.path.append("../../")

from models.conditional_ktrees_model.conditional_ktrees_model import ConditionalKTreesModel
from thex_data.data_consts import TARGET_LABEL, class_to_subclass, UNDEF_CLASS
from thex_data.data_clean import *


import warnings


"""

Tests ConditionalKTreesModel

Run tests with:
python -m unittest

"""


class TestConditionalKTreesModel(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')
        self.test_model = ConditionalKTreesModel()
        hierarchy = class_to_subclass
        for parent in class_to_subclass.keys():
            hierarchy[parent].append(UNDEF_CLASS + parent)
        self.test_model.tree = init_tree(hierarchy)
        self.test_model.class_levels = assign_levels(
            self.test_model.tree, {}, self.test_model.tree.root, 1)
        self.test_model.min_class_size = 1

    def test_train(self):
        self.test_model.y_train = pd.DataFrame([
            ["Ia"], ["CC"], ["CC"], ["Ia"], ["Ia"]], columns=[TARGET_LABEL])
        self.test_model.X_train = pd.DataFrame([
            [20, 15], [5, 8], [6, 8], [21, 16], [18, 16]], columns=['feature1', 'feature2'])

        subclassifiers = self.test_model.train()
        self.assertEqual(len(subclassifiers.keys()), 1)
        self.assertEqual(set(subclassifiers['TTypes'].classes), set(['CC', 'Ia']))

    def test_get_subclf_classes(self):
        self.test_model.y_train = pd.DataFrame([
            ["Ia"], ["CC"], ["CC"], ["Ia-91bg"], ["Ia"]], columns=[TARGET_LABEL])
        self.test_model.X_train = pd.DataFrame([
            [20, 15], [5, 8], [6, 8], [21, 16], [18, 16]], columns=['feature1', 'feature2'])

        classes = self.test_model.get_subclf_classes(
            child_classes=["Ia-91bg"], y=self.test_model.y_train, parent_class="Ia")
        print(classes)
        self.assertEqual(set(classes), set(["Ia-91bg", UNDEF_CLASS + "Ia"]))

    def test_get_subclf_data(self):
        y = pd.DataFrame([
            ["Ia"], ["CC"], ["CC"], ["Ia, Ia-91bg"], ["Ia"]], columns=[TARGET_LABEL])
        X = pd.DataFrame([
            [20, 15], [5, 8], [6, 8], [21, 16], [18, 16]], columns=['feature1', 'feature2'])
        class_labels = ['Ia', 'CC']

        X, y = self.test_model.get_subclf_data(X, y, class_labels)
        exp_X = pd.DataFrame([
            [20, 15], [5, 8], [6, 8], [21, 16], [18, 16]], columns=['feature1', 'feature2'])
        exp_y = pd.DataFrame([
            [0], [1], [1], [0], [0]], columns=[TARGET_LABEL])
        exp_X.equals(X)
        exp_y.equals(y)
