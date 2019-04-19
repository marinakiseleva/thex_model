

class KTreeModel(BaseModel):
    """
    Model that classifies using unique Kernel Density Estimates for distributions of each feature, of each class. 
    """

    def __init__(self, cols=None, col_matches=None, **data_args):
        self.name = "K-Tree Model"
        self.naive = data_args['naive'] if 'naive' in data_args else False
        self.cols = cols
        self.col_matches = col_matches
        self.user_data_filters = data_args

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """

    def test_model(self):
        predicted_classes = self.test()
        return predicted_classes

    def get_class_probabilities(self, x):
        return self.calculate_class_probabilities(x)
