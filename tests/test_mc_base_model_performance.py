import sys
import unittest
import pandas as pd
sys.path.append("..")
import numpy as np

from models.base_model_mc.mc_base_model_performance import MCBaseModelPerformance
# from models.mc_kde_model.mc_kde_model import *
from thex_data.data_consts import TARGET_LABEL, PRED_LABEL

"""

Tests MCKDEModel

Run tests with:
python -m unittest

"""


class TestMCKDEModelLogic(unittest.TestCase):

    def test_combine_mc_pred_actual(self):
        test_model = MCBaseModelPerformance()
        class_names = ["I", "Ia", "II", "II P", "Undefined_II P", "Undefined_II"]
        test_model.test_level = 2
        test_model.class_levels = {"I": 1, "II": 1, "II P": 2, "Ia": 2}
        test_model.predictions = pd.DataFrame(
            [["Ia"], ["II P"], ["I"]], columns=[PRED_LABEL])
        test_model.y_test = pd.DataFrame(
            [["I, Ia"], ["II, II P"], ["II, II P"]], columns=[TARGET_LABEL])
        combined = test_model.combine_mc_pred_actual(class_names)

        expected_data = [["Ia",   1, 1, 0, 0, 0, 0],
                         ["II P", 0, 0, 1, 1, 1, 0],
                         ["I",    0, 0, 1, 1, 1, 0], ]
        expected = pd.DataFrame(expected_data, columns=[
                                "predicted_class",  "I",  "Ia", "II",  "II P", "Undefined_II P", "Undefined_II"])

        self.assertTrue(combined.equals(expected))

    def test_get_mc_class_performance(self):
        test_model = MCBaseModelPerformance()
        class_names = ["Ia", "Undefined_II",  "II P"]
        test_model.test_level = 2
        test_model.class_levels = {"I": 1, "II": 1, "II P": 2, "Ia": 2}
        test_model.predictions = pd.DataFrame(
            [["Ia"], ["II P"], ["I"]], columns=[PRED_LABEL])
        test_model.y_test = pd.DataFrame(
            [["I, Ia"], ["II, II P"], ["II, II P"]], columns=[TARGET_LABEL])
        actual = test_model.get_mc_class_performance(class_names)
        # [AP, TP, FP]
        expected = {"II P": [2, 1, 0],
                    "Undefined_II": [0, 0, 0],
                    "Ia": [1, 1, 0]
                    }
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
