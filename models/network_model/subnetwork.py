from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from thex_data.data_consts import TARGET_LABEL, class_to_subclass
from thex_data.data_clean import convert_class_vectors


def create_model(learn_rate=0.1, momentum=0.1, decay=0.1, nesterov=False, input_length=1, output_length=1):
    model = Sequential()
    model.add(Dense(48, input_dim=input_length, activation='relu'))
    # model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_length, activation='softmax'))

    sgd = optimizers.SGD(lr=learn_rate, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy', 'categorical_accuracy'])
    return model


class SubNetwork:

    def __init__(self, classes, X, y):
        """
        Neural network with first layer of length input_length, last layer of output_length, where each output node corresponds to a class in classes
        :param classes: List of classes this network runs on; each label in y coresponds to index in this list
        :param y: DataFrame of labels for these classes
        :param X: DataFrame of features for these classes
        """
        self.input_length = len(list(X))
        self.output_length = len(classes)
        self.classes = classes

        print("\n\nInitialzing SubNetwork for classes " + str(self.classes))
        self.network = self.init_network(X, y)

    def get_sample_weights(self, X, y):
        """
        Get weight of each sample (1/# of samples in class)
        """
        labeled_samples = pd.concat([X, y], axis=1)
        classes = labeled_samples[TARGET_LABEL].unique()
        label_counts = {}
        for c in classes:
            label_counts[c] = labeled_samples.loc[
                labeled_samples[TARGET_LABEL] == c].shape[0]
        sample_weights = []
        for df_index, row in labeled_samples.iterrows():
            class_count = label_counts[row[TARGET_LABEL]]
            sample_weights.append(1 / class_count)
        return np.array(sample_weights)

    def init_network(self, X, y):

        # Initialize hyperparameters
        # SGD
        learning_rate = 0.001
        momentum = 0.5  # direction of the next step with the knowledge of the previous steps
        decay = 0.01  # regularization weight lambda
        # NN
        epochs = 500
        batch_size = 24

        # Create NN
        model = Sequential()
        model.add(Dense(48, input_dim=self.input_length, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.output_length, activation='softmax'))
        sgd = optimizers.SGD(lr=learning_rate, momentum=momentum,
                             decay=decay, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy', 'categorical_accuracy'])

        # Split data into Training and Validation
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.25, shuffle=True)
        val_sample_weights = self.get_sample_weights(x_valid, y_valid)
        train_sample_weights = self.get_sample_weights(x_train, y_train)

        # Convert numeric labels to one-hot encoding (which is what Keras expects)
        y_train_codes = to_categorical(y_train.values)
        y_val_codes = to_categorical(y_valid.values)

        # Assign class weights
        class_indices = list(range(len(self.classes)))
        class_weights = compute_class_weight(
            class_weight='balanced', classes=class_indices, y=y[TARGET_LABEL].values)
        d_class_weights = dict(enumerate(class_weights))

        es = EarlyStopping(monitor='val_loss',  mode='min', verbose=1,
                           patience=20)
        model.fit(x_train.values, y_train_codes, batch_size=batch_size, verbose=0, epochs=epochs, sample_weight=train_sample_weights,
                  validation_data=(x_valid.values, y_val_codes, val_sample_weights), callbacks=[es])
        # # , class_weight=d_class_weights)  # , callbacks=[es]
        metrics = model.evaluate(x_valid.values, y_val_codes)
        print("Model metrics on valiation set: ")
        print(model.metrics_names)
        print(metrics)
        # model = self.get_best_model(X, y, batch_size, epochs)
        return model

    def get_best_model(self, X, y, batch_size, epochs):
        # Grid search
        learn_rates = [0.0001, 0.001, 0.01, 0.1, 0.2]
        momentums = [0.0, 0.2, 0.5, 0.7, 0.9]  # usually between 0.5 to 0.9
        decays = [0.0,  0.2,  0.6]

        first_layer_size = [40, 80, 120]
        second_layer_size = [20, 34, 60]

        # Best: 0.678211 using {'decay': 0.2, 'input_length': 484, 'learn_rate': 0.01, 'momentum': 0.7, 'nesterov': False, 'output_length': 2}
        # Best: 0.895062 using {'decay': 0.0, 'input_length': 484, 'learn_rate':
        # 0.0001, 'momentum': 0.2, 'nesterov': True, 'output_length': 5}
        param_grid = dict(learn_rate=learn_rates,
                          momentum=momentums, decay=decays, nesterov=[True, False],
                          output_length=[self.output_length], input_length=[self.input_length])

        grid_model = KerasClassifier(build_fn=create_model, epochs=epochs,
                                     batch_size=batch_size, verbose=1)
        grid = GridSearchCV(estimator=grid_model,
                            param_grid=param_grid, n_jobs=-1, verbose=1, cv=3)

        # sample_weight=train_sample_weights,
        es = EarlyStopping(monitor='val_loss',  mode='min', verbose=1,
                           patience=20)
        y_onehot = to_categorical(y.values)
        sample_weights = self.get_sample_weights(X, y)
        grid.fit(X.values, y_onehot, sample_weight=sample_weights, callbacks=[es])
        print("\nBest model params")
        print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

        return grid.best_estimator_
