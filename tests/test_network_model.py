import sys
import unittest
import random

import numpy as np
import pandas as pd
sys.path.append("..")
from hmc import hmc

from models.network_model.network_model import *
from thex_data.data_consts import TARGET_LABEL

"""

Tests network_model

Run tests with:
python -m unittest

"""


class TestNetwork(unittest.TestCase):

    def setUp(self):
        y = pd.DataFrame([
            [0], [1], [1], [0], [0]], columns=[TARGET_LABEL])
        X = pd.DataFrame([
            [20, 15], [5, 8], [6, 8], [21, 16], [18, 16]], columns=['feature1', 'feature2'])
        classes = [0, 1]
        random.seed(2)
        self.test_model = NetworkModel()
        self.test_model.X_train = X
        self.test_model.y_train = y

        hmc_hierarchy = hmc.ClassHierarchy("TTypes")
        hmc_hierarchy.add_node(1, "TTypes")
        hmc_hierarchy.add_node(1, 0)
        self.test_model.tree = hmc_hierarchy

        self.test_model.class_levels = {0: 0, 1: 1}
        self.test_model.train()

    def test_subnet(self):
        test_sample = np.array([20, 16])

        # self.assertEqual(round(probabilities['class2'], 2), 0.39)
        predictions = self.test_model.get_class_probabilities(
            x=test_sample)
        print(predictions)


if __name__ == '__main__':
    unittest.main()
