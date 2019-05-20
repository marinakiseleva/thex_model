import numpy as np

from models.base_model_mc.mc_base_model import MCBaseModel
from models.network_model.network_train import NetworkTrain

from thex_data.data_consts import cat_code


class NetworkModel(MCBaseModel, NetworkTrain):
    """
    Model that consists of K-trees, where is K is total number of all unique class labels (at all levels of the hierarchy). Each sample is given a probability of each class, using each tree separately. Thus, an item could have 90% probability of I and 99% of Ia.
    """

    def __init__(self, **data_args):
        self.name = "Neural Network Model"
        data_args['transform_labels'] = False
        self.user_data_filters = data_args
        self.networks = {}

    def train_model(self):
        """
        Train K-trees, where K is the total number of classes in the data (at all levels of the hierarchy)
        """
        if self.class_labels is None:
            self.class_labels = self.get_mc_unique_classes(self.y_train)
        return self.train()

    def test_model(self):
        """
        Get class prediction for each sample.
        """
        raise ValueError("test_model not impelmented")

    def get_all_class_probabilities(self):
        """
        Get class probability for each sample in self.X_test
        :return probabilities: Numpy Matrix with each row corresponding to sample, and each column the probability of that class, in order of self.class_labels
        """
        print("all class labels are")
        print(self.class_labels)
        # Start from root (of class hierarchy) network
        root_subnet = self.networks[self.tree.root]
        predictions = []
        for df_index, x in self.X_test.iterrows():
            prob_map = {c: 0 for c in self.class_labels}
            prob_map = self.get_class_probabilities(
                np.array([x.values]), root_subnet, prob_map)
            predictions.append(list(prob_map.values()))
        print("Final output")
        print(np.array(predictions))
        return np.array(predictions)

    def get_class_probabilities(self, x, subnet, probabilities):
        """
        Calculates probability of each transient class for the single test data point (x). 
        :param x: Single row of features 
        :param subnet: Current SubNetwork to run test point through. Start at root.
        :return: map from class_names to probabilities
        """

        predictions = subnet.network.predict(x=x,  batch_size=1, verbose=0)
        predictions = list(predictions[0])
        max_class_index = predictions.index(max(predictions))
        max_prob_class = subnet.classes[max_class_index]

        # Update probabilities
        for pred_index, pred_class_prob in enumerate(predictions):
            pred_class_name = subnet.classes[pred_index]
            probabilities[pred_class_name] = pred_class_prob

        if max_prob_class in self.networks:
            next_net = self.networks[max_prob_class]
            return self.get_class_probabilities(x, next_net, probabilities)
        else:
            return probabilities
