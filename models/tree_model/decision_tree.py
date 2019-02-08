from sklearn import tree


def run_tree(data_columns, incl_redshift=False):
    X_train, X_test, y_train, y_test = get_train_test(
        data_columns, incl_redshift, cat=True)
    # Test on training
    # X_test = X_train
    # y_test = y_train

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',
                                      max_depth=30, min_samples_split=10, min_samples_leaf=4, random_state=20)
    clf.fit(X_train, X_test)

    predictions = tree.predict(X_test)
    dth_accuracy = tree.score(X_test, y_test)

    return predictions, y_test
    # tree = train_model(tree, X_train, X_test)
    # evaluate_model(tree, hmc_hierarchy, X_test, y_test)

    # get_feature_importance(clf, train)

    # # print("Testing on Training Data...")
    # # test = train.copy()

    # predictions = clf.predict(test.drop([TARGET_LABEL], axis=1).values)

    # actual_classes = test[[TARGET_LABEL]].values
    # plot_confusion_matrix(actual_classes, predictions,
    #                       normalize=True,
    #                       title='Confusion matrix',
    #                       cmap=plt.cm.Blues)
    # pd_predictions = pd.DataFrame(predictions.astype(int), columns=['predicted_class'])
    # pd_actual = test[[TARGET_LABEL]].astype(int).reset_index(drop=True)
    # compute_plot_class_accuracy(
    # pd_predictions, pd_actual, plot_title="Tree Accuracy per Class, on
    # Testing Data")
