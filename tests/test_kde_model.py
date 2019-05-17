import sys
import unittest
import pandas as pd
sys.path.append("..")
import numpy as np

from models.kde_model.kde_model import *
from thex_data.data_consts import TARGET_LABEL

"""

Tests BaseModel logic through KDE model.

Run tests with:
python -m unittest 

"""


class TestKDEModelLogic(unittest.TestCase):

    def setUp(self):
        self.test_kdemodel = KDEModel()  # Test instance
        # DataFrame that represents labels (y), each row is class vector
        self.test_df = pd.DataFrame(
            [[[1, 0, 1]], [[0, 0, 1]], [[0, 0, 1]]], columns=[TARGET_LABEL])
        self.full_test_df = pd.DataFrame(
            [[4, 6, [1, 0, 1]], [1, 3, [0, 0, 1]], [8, 9, [0, 0, 1]]], columns=['feature1', 'feature2', TARGET_LABEL])

    # def test_get_probability_matrix(self):
        """
        get_probability_matrix(self, class_code=None)
        """
        # Test with class_code
        # self.test_kdemodel.y_test
        # self.test_kdemodel.X_test
        # self.test_kdemodel.get_probability_matrix()

    def test_aggregate_prob_ranges(self):
        """ 
        test aggregate_prob_ranges
        """
        percent_ranges = ['5%', '15%', '25%', '35%',
                          '45%', '55%', '65%', '75%', '85%', '95%']
        corr_in_range1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        count_in_range1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        prob_ranges1 = [percent_ranges, corr_in_range1, count_in_range1]

        c2_corr1 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
        c2_count1 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
        c2_corr2 = [0, 0, 0, 0, 0, 2, 0, 0, 1, 0]
        c2_count2 = [0, 0, 0, 0, 0, 3, 0, 0, 1, 0]
        c2_1 = [percent_ranges, c2_corr1, c2_count1]
        c2_2 = [percent_ranges, c2_corr2, c2_count2]
        d = {'class1': [prob_ranges1], 'class2': [c2_1, c2_2]}

        a = self.test_kdemodel.aggregate_prob_ranges(d)
        self.assertEqual(a['class2'][1], [0, 0, 0, 0, 0, 3, 0, 0, 2, 0])
        self.assertEqual(a['class2'][2], [0, 0, 0, 0, 0, 4, 0, 0, 2, 0])

    def test_get_corr_prob_ranges_all_classes(self):
        """
        Tests get_corr_prob_ranges when using an X_accs of ALL classes
        """

        """
        class1  class2  ... classN      actual_class    predicted_class
        .88        .1       .12             class2      class2
        """
        # TEST 1 : 0/1 correct
        X_accs = pd.DataFrame([[.8, .1, .1, 'class1', 'class2']], columns=[
                              'class1', 'class2', 'class3', 'actual_class', 'predicted_class'])
        percent_ranges, corr_ranges, count_ranges = self.test_kdemodel.get_corr_prob_ranges(
            X_accs, 'class1')
        self.assertEqual(percent_ranges, [
                         '5%', '15%', '25%', '35%', '45%', '55%', '65%', '75%', '85%', '95%'])
        self.assertEqual(corr_ranges, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(count_ranges, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

        # TEST 2: 1/1 correct
        X_accs = pd.DataFrame([[.4, .3, .3, 'class1', 'class1']], columns=[
                              'class1', 'class2', 'class3', 'actual_class', 'predicted_class'])
        percent_ranges, corr_ranges, count_ranges = self.test_kdemodel.get_corr_prob_ranges(
            X_accs, 'class1')
        self.assertEqual(corr_ranges, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(count_ranges, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    def test_aggregate_accuracies(self):
        """
        Test function aggregate_accuracies(self, model_results, unique_classes)
        """
        unique_classes = [1, 2, 3]
        dict1 = {1: 1, 2: .5, 3: 0}
        dict2 = {1: 1, 2: 1, 3: .5}
        model_results = [dict1, dict2]
        d = self.test_kdemodel.aggregate_accuracies(model_results, unique_classes)
        self.assertEqual(round(d[1], 2), 1.00)
        self.assertEqual(round(d[2], 2), 0.75)
        self.assertEqual(round(d[3], 2), 0.25)

        # Ensure aggregate changes nothing if there is only 1 dict
        d = self.test_kdemodel.aggregate_accuracies([dict1], unique_classes)
        self.assertEqual(round(d[1], 2), 1.00)
        self.assertEqual(round(d[2], 2), 0.50)
        self.assertEqual(round(d[3], 2), 0)

    def test_get_class_precisions(self):
        """
        Test function get_class_precisions
        """
        self.test_kdemodel.predictions = [1, 1, 1, 2, 3]
        self.test_kdemodel.y_test = [1, 1, 2, 2, 1]
        d = self.test_kdemodel.get_class_precisions()
        self.assertEqual(round(d[1], 2), 0.67)
        self.assertEqual(round(d[2], 2), 1.00)
        self.assertEqual(round(d[3], 2), 0.00)

    def test_get_class_precision(self):
        """
        """
        self.test_kdemodel.predictions = [1, 1, 1, 2, 3]
        self.test_kdemodel.y_test = [1, 1, 2, 2, 1]

        df = self.test_kdemodel.combine_pred_actual()

        class_precision = self.test_kdemodel.get_class_precision(
            df_compare=df, class_code=1)
        self.assertEqual(round(class_precision, 2), 0.67)

        class_precision = self.test_kdemodel.get_class_precision(
            df_compare=df, class_code=2)
        self.assertEqual(class_precision, 1)

        class_precision = self.test_kdemodel.get_class_precision(
            df_compare=df, class_code=3)
        self.assertEqual(class_precision, 0)


if __name__ == '__main__':
    unittest.main()
