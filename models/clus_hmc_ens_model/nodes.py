class LeafNode:

    def __init__(self, guess):
        self.guess = guess


class InternalNode:

    def __init__(self, feature_val_pair, sample_greater, sample_less):
        """
        Initialize internal node with 2 children: sample_greater is the child with values >= value for the feature, and sample_less is opposite.
        """
        self.feature = feature_val_pair[0]
        self.feature_value = feature_val_pair[1]
        self.sample_greater = sample_greater
        self.sample_less = sample_less
