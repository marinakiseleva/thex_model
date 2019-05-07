class LeafNode:

    def __init__(self, guess):
        """
        Leaf node maintains a guess for this node, which is the predicted label for any sample in the Leaf. Each value in vector is the ratio of the predicted class out of the total number of samples in this leaf.
        :param guess: prediction for leaf; average class vector, with each value as probability of class. Order is the same as self.class_labels
        """
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
