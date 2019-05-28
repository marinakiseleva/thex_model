import sys
import unittest
import random

import numpy as np
import pandas as pd
sys.path.append("..")


from models.network_model.subnetwork import *
from thex_data.data_consts import TARGET_LABEL

"""

Tests Subnetwork

Run tests with:
python -m unittest

"""


class TestSubnetwork(unittest.TestCase):

    def setUp(self):
        y = pd.DataFrame([
            [0], [1], [1], [0], [0]], columns=[TARGET_LABEL])
        X = pd.DataFrame([
            [20, 15], [5, 8], [6, 8], [21, 16], [18, 16]], columns=['feature1', 'feature2'])
        classes = [0, 1]
        random.seed(2)
        self.test_model = SubNetwork(classes, X, y)

    def test_subnet(self):
        test_sample = np.array([20, 16])

        # self.assertEqual(round(probabilities['class2'], 2), 0.39)
        predictions = self.test_model.network.predict(
            x=test_sample,  batch_size=1, verbose=0)
        predictions = list(predictions[0])
        max_class_index = predictions.index(max(predictions))
        max_prob_class = self.test_model.classes[max_class_index]
        print("max prob class " + str(max_prob_class))


if __name__ == '__main__':
    unittest.main()
