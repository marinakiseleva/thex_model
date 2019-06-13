from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

from thex_data.data_consts import TARGET_LABEL
from models.conditional_model.classifier import SubClassifier


def create_model(layer_sizes=[48, 16], input_length=1, output_length=1,
                 learn_rate=0.1, momentum=0.1, decay=0.1, nesterov=False):
    model = Sequential()
    # Add fully connected layers
    model.add(Dense(layer_sizes[0], input_dim=input_length, activation='relu'))
    if len(layer_sizes) > 2:
        # Add more layers, as specified
        for layer_size in layer_sizes[1:-1]:
            model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(layer_sizes[-1], activation='relu'))
    model.add(Dense(output_length, activation='softmax'))

    # sgd = optimizers.SGD(lr=learn_rate, momentum=momentum,
    #                      decay=decay, nesterov=nesterov)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',  # sgd,  # 'adam'
                  metrics=['accuracy', 'categorical_accuracy'])
    return model


class SubNetwork(SubClassifier):

    def __init__(self, classes, X, y):
        """
        Neural network with first layer of length input_length, last layer of output_length, where each output node corresponds to a class in classes
        :param classes: List of classes this network runs on; each label in y coresponds to index in this list
        :param X: DataFrame of features for these classes
        :param y: DataFrame of labels for these classes, as values
        """
        self.input_length = len(list(X))
        self.output_length = len(classes)
        super(SubNetwork, self).__init__(classes, X, y)

    def predict(self, x):
        """
        Return probabilities as list, in same order as self.classes
        :param x: 2D Numpy array of features for single row
        """
        return list(self.classifier.predict(x=x,  batch_size=1)[0])

    def init_classifier(self, X, y):
        """
        Create Keras Neural Network for this set of data. Use sample weights and grid search to create optimal network.
        :param X: DataFrame of features
        :param y: DataFrame of numeric labels, each number corresponding to index in self.classes
        """
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.5)
        weights_train = self.get_sample_weights(y_train)
        weights_valid = self.get_sample_weights(y_valid)
        # Convert numeric labels to one-hot encoding (which is what Keras fit expects)
        y_train = to_categorical(y=y_train, num_classes=len(self.classes))
        y_valid = to_categorical(y=y_valid, num_classes=len(self.classes))

        # Assign class weights
        class_indices = list(range(len(self.classes)))
        class_weights = compute_class_weight(
            class_weight='balanced', classes=class_indices, y=y[TARGET_LABEL].values)
        class_weights = dict(enumerate(class_weights))

        # NN hyperparameters
        epochs = 1500
        batch_size = 24
        es = EarlyStopping(monitor='val_loss',
                           mode='min',
                           verbose=0,
                           min_delta=0,
                           patience=10,
                           restore_best_weights=True)

        model = self.get_best_model(batch_size=batch_size,
                                    epochs=epochs,
                                    x_valid=x_valid.values,
                                    y_valid=y_valid,
                                    val_sample_weights=weights_valid,
                                    x_train=x_train.values,
                                    y_train=y_train,
                                    train_sample_weights=weights_train,
                                    callbacks=[es],
                                    class_weights=class_weights)

        metrics = model.fit(x_train.values,
                            y_train,
                            batch_size=batch_size,
                            verbose=0,
                            epochs=epochs,
                            sample_weight=weights_train,
                            validation_data=(x_valid.values, y_valid,
                                             weights_valid),  # , weights_valid
                            # class_weight=class_weights,
                            callbacks=[es]
                            )

        print("Validation loss: "
              + str(metrics.history['val_loss'][-1])
              + " , training loss: " + str(metrics.history['loss'][-1]))
        metrics = model.evaluate(x_valid.values, y_valid)
        print("Valiation " +
              str(model.metrics_names[2]) + " : " + str(round(metrics[2], 4)))
        print("Finished at epoch " + str(es.stopped_epoch))
        if es.stopped_epoch == 0:
            print("Early stopping did not take effect, validation loss never increased. Simply ended at max of " + str(epochs) + "epochs.")

        return model

    def get_best_model(self, batch_size, epochs, x_valid, y_valid, val_sample_weights, x_train, y_train, train_sample_weights, callbacks, class_weights):

        param_grid = {
            # 'learn_rate': [0.0001, 0.001],  # , 0.01
            # 'momentum': [0.5],  # usually between 0.5-0.9
            # 'decay': [0.2, 0.6],
            # 'nesterov': [True],  # False
            # , [88, 64, 16], [64, 38, 24, 12]],[88, 44, 32, 16]
            'layer_sizes': [[92, 32], [108, 64, 16]],
            'input_length': [self.input_length],
            'output_length': [self.output_length]}

        grid_model = KerasClassifier(build_fn=create_model,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     verbose=0)

        keras_fit_params = {
            # 'callbacks': callbacks,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_data': (x_valid, y_valid),  # , val_sample_weights
            'verbose': 0
        }
        grid = GridSearchCV(estimator=grid_model,
                            param_grid=param_grid,
                            fit_params=keras_fit_params,
                            n_jobs=-1,
                            verbose=0,
                            cv=3)

        # , class_weight=class_weights)
        grid.fit(x_train, y_train, shuffle=True, sample_weight=train_sample_weights)
        # sample_weight=train_sample_weights)

        m = create_model(
            # learn_rate=grid.best_params_['learn_rate'],
            # momentum=grid.best_params_['momentum'],
            # decay=grid.best_params_['decay'],
            # nesterov=grid.best_params_['nesterov'],
            layer_sizes=grid.best_params_['layer_sizes'],
            input_length=grid.best_params_['input_length'],
            output_length=grid.best_params_['output_length'])
        print("Best model parameters")
        print(grid.best_params_)

        return m
