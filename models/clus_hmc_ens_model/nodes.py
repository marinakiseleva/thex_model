class LeafNode:

    def __init__(self, guess):
        self.guess = guess


class InternalNode:

    def __init__(self, feature_val_pair, sample_without, sample_with):
        """
        Initialize internal node with 2 children: sample_without is the child that does not have the feature/value pair, and sample_with does
        """
        self.feature_val_pair = feature_val_pair
        self.sample_without = sample_without
        self.sample_with = sample_with

    # def classify(self, features):
    #     """
    #     Return most likely label using features
    #     """
    #     return test(self, features)
