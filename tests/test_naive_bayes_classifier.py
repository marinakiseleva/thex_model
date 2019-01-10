import sys
import unittest
sys.path.append("..")

from thex_bayes import nb_classifier


class TestNBFunctions(unittest.TestCase):

    def test_mean(self):
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        real_mean = 5.5
        func_mean = nb_classifier.mean(nums)
        self.assertEqual(real_mean, func_mean)

    def test_std_dev(self):
        nums = [2, 3]
        real_val = .5
        func_val = nb_classifier.stdev(nums)
        self.assertEqual(real_val, func_val)

    def test_calculate_probability(self):
        real_val = .399
        func_val = nb_classifier.calculate_probability_density(3, 3, 1)
        self.assertEqual(real_val, round(func_val, 3))


if __name__ == '__main__':
    unittest.main()
