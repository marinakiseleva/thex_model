import argparse
from sklearn import tree

from thex_data.data_init import *
from thex_data.data_prep import get_train_test, get_data
from thex_data.data_consts import code_cat, TARGET_LABEL

from model_performance.performance import *


def decision_tree(data_columns, incl_redshift=False):

    train, test = get_train_test(col_list=data_columns, incl_redshift=incl_redshift)

    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best',
                                      max_depth=30, min_samples_split=10, min_samples_leaf=4, random_state=20)

    X = train.drop([TARGET_LABEL], axis=1)
    y = train[[TARGET_LABEL]]
    clf = clf.fit(X.values, y.values)

    get_feature_importance(clf, train)

    # print("Testing on Training Data...")
    # test = train.copy()

    predictions = clf.predict(test.drop([TARGET_LABEL], axis=1).values)

    actual_classes = test[[TARGET_LABEL]].values
    plot_confusion_matrix(actual_classes, predictions,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)
    pd_predictions = pd.DataFrame(predictions.astype(int), columns=['predicted_class'])
    pd_actual = test[[TARGET_LABEL]].astype(int).reset_index(drop=True)
    compute_plot_class_accuracy(
        pd_predictions, pd_actual, plot_title="Tree Accuracy per Class, on Testing Data")


def main():
    parser = argparse.ArgumentParser(description='Classify transients')
    parser.add_argument('-cols', '--cols', nargs='+',
                        help='<Required> Pass column names', required=False)
    parser.add_argument('-col_names', '--col_names', nargs='+',
                        help='Pass in string by which columns will be selected. For example: AllWISE will use all AlLWISE columns.', required=False)
    parser.add_argument('-incl_redshift', '--incl_redshift', nargs='+',
                        help='<Required> Set flag', required=False)
    args = parser.parse_args()

    col_list = []
    if args.cols is None:
        if args.col_names is not None:
            col_list = collect_cols(
                "./../../../data/THEx-catalog.v0_0_3.fits", args.col_names)
        else:
            # Use all columns in data set with number values
            cols = list(collect_data("./../../../data/THEx-catalog.v0_0_3.fits"))
            drop_cols = ['event', 'ra', 'dec', 'ra_deg', 'dec_deg', 'radec_err', 'redshift', 'claimedtype', 'host', 'host_ra', 'host_dec', 'ebv', 'host_ra_deg',
                         'host_dec_deg', 'host_dist', 'host_search_radius', 'is_confirmed_host', 'by_primary_cand', 'by_transient', "Err", "_e_", 'AllWISE_IsVar',
                         'HyperLEDA_objtype',
                         'HyperLEDA_type',
                         'HyperLEDA_bar',
                         'HyperLEDA_ring',
                         'HyperLEDA_multiple',
                         'HyperLEDA_compactness',
                         'HyperLEDA_agnclass'
                         ]
            col_list = []
            for c in cols:
                if not any(drop in c for drop in drop_cols):
                    col_list.append(c)

    else:
        col_list = args.cols

    incl_redshift = args.incl_redshift if args.incl_redshift is not None else False

    print("\n\nUsing data columns:\n\n" + str(col_list))

    decision_tree(col_list, incl_redshift=args.incl_redshift)


if __name__ == '__main__':
    main()
