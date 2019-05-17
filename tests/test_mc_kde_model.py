import sys
import unittest
import pandas as pd
sys.path.append("..")
import numpy as np

from sklearn.neighbors.kde import KernelDensity

from models.mc_kde_model.mc_kde_model import *
from thex_data.data_consts import TARGET_LABEL

"""

Tests MCKDEModel

Run tests with:
python -m unittest

"""


class TestMCKDEModelLogic(unittest.TestCase):

    def setUp(self):
        self.test_kdemodel = MCKDEModel()  # Test instance
        self.full_test_df = pd.DataFrame(
            [[4, 6, 2],
             [5, 5, 2],
             [1, 3, 1],
             [8, 9, 1]],
            columns=['feature1', 'feature2', TARGET_LABEL])

        model1 = KernelDensity()
        model1.fit(X=[[1, 3], [8, 9]], y=None)

        model2 = KernelDensity()
        model2.fit(X=[[4, 6], [5, 5]], y=None)
        self.test_kdemodel.models = {"class1": model1, "class2": model2}

        self.test_kdemodel.class_labels = ["class1", "class2"]

    def test_get_class_probabilities(self):
        x = pd.Series([5, 9])
        probabilities = self.test_kdemodel.get_class_probabilities(x)

        self.assertEqual(round(probabilities['class1'], 2), 0.61)
        self.assertEqual(round(probabilities['class2'], 2), 0.39)

    def test_get_all_class_probabilities(self):
        self.test_kdemodel.X_test = pd.DataFrame(
            [[5, 9], [1, 1]], columns=['feature1', 'feature2'])
        probabilities = self.test_kdemodel.get_all_class_probabilities()
        self.assertEqual(round(probabilities[0][0], 2), 0.61)
        self.assertEqual(round(probabilities[0][1], 2), 0.39)
        self.assertEqual(round(probabilities[1][0], 2), 1)
        self.assertEqual(round(probabilities[1][1], 2), 0)

if __name__ == '__main__':
    unittest.main()
