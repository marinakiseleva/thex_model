import math
import random
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier


from thex_data.data_consts import TARGET_LABEL, class_to_subclass
from thex_data.data_clean import convert_class_vectors


def create_model(learn_rate=0.1, momentum=0.1, decay=0.1, nesterov=False, layer_sizes=[48, 16], input_length=1, output_length=1):

    model = Sequential()
    # Add fully connected layers
    model.add(Dense(layer_sizes[0], input_dim=input_length, activation='relu'))
    if len(layer_sizes) > 2:
        # Add more layers, as specified
        for layer_size in layer_sizes[1:-1]:
            model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(layer_sizes[-1], activation='relu'))
    model.add(Dense(output_length, activation='softmax'))

    sgd = optimizers.SGD(lr=learn_rate, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy', 'categorical_accuracy'])
    return model


class SubNetwork:

    def __init__(self, classes, X, y):
        """
        Neural network with first layer of length input_length, last layer of output_length, where each output node corresponds to a class in classes
        :param classes: List of classes this network runs on; each label in y coresponds to index in this list
        :param X: DataFrame of features for these classes
        :param y: DataFrame of labels for these classes, as values
        """
        self.input_length = len(list(X))
        self.output_length = len(classes)
        self.classes = classes

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

        sample_weights = self.get_sample_weights(X, y)

        # Convert numeric labels to one-hot encoding (which is what Keras expects)
        y_vectors = to_categorical(y)

        validation_count = math.ceil(X.shape[0] * 0.25)
        validation_indices = random.sample(range(X.shape[0]), validation_count)

        # Split data into Training and Validation
        # TODO: update weights so they are computed separately for training and
        # validation, otherwise training weights will be skewed.
        x_valid = X[X.index.isin(validation_indices)].values
        y_valid = np.take(y_vectors, validation_indices, axis=0)
        val_sample_weights = np.take(sample_weights, validation_indices, axis=0)
        x_train = X[~X.index.isin(validation_indices)].values
        y_train = np.delete(y_vectors, validation_indices, axis=0)
        train_sample_weights = np.delete(sample_weights, validation_indices, axis=0)

        # Assign class weights
        class_indices = list(range(len(self.classes)))
        class_weights = compute_class_weight(
            class_weight='balanced', classes=class_indices, y=y[TARGET_LABEL].values)

        es = EarlyStopping(monitor='val_loss',  mode='auto', verbose=0,
                           patience=20, restore_best_weights=True)
        # NN hyperparameters
        epochs = 500
        batch_size = 24
        model = self.get_best_model(X, y_vectors, sample_weights, batch_size, epochs)

        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  verbose=0,
                  epochs=epochs,
                  sample_weight=train_sample_weights,
                  validation_data=(x_valid, y_valid, val_sample_weights),
                  # class_weight=dict(enumerate(class_weights)),
                  callbacks=[es]
                  )

        metrics = model.evaluate(x_valid, y_valid)
        print("Valiation " +
              str(model.metrics_names[2]) + " : " + str(round(metrics[2], 4)))

        return model

    def get_best_model(self, X, y_vectors, sample_weights, batch_size, epochs):

        param_grid = {'learn_rate': [0.001],  # , 0.01, 0.1],
                      'momentum': [0.5],  # , 0.7, 0.9]  # usually between 0.5 to 0.9,
                      'decay': [0.0,  0.2,  0.6],
                      'nesterov': [True, False],
                      'layer_sizes': [[48, 16]],  # , [48, 32, 16], [64, 38, 24, 12]],
                      'input_length': [self.input_length],
                      'output_length': [self.output_length]}

        grid_model = KerasClassifier(build_fn=create_model,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=0)

        grid = GridSearchCV(estimator=grid_model,
                            param_grid=param_grid,
                            n_jobs=-1,
                            verbose=0,
                            cv=3)
        grid.fit(X.values,
                 y_vectors,
                 sample_weight=sample_weights)

        m = create_model(learn_rate=grid.best_params_['learn_rate'],
                         momentum=grid.best_params_['momentum'],
                         decay=grid.best_params_['decay'],
                         nesterov=grid.best_params_['nesterov'],
                         layer_sizes=grid.best_params_['layer_sizes'],
                         input_length=grid.best_params_['input_length'],
                         output_length=grid.best_params_['output_length'])

        return m
