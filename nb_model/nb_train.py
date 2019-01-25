from thex_data.data_consts import TARGET_LABEL
import scipy.stats as stats

"""
Logic for training the Naive Bayes classifier
"""


def find_best_fitting_dist(data):
    """
    Finds best fitting distribution for this particular set of features (features of a single transient type)
    """
    distributions = [stats.norm]
    mles = []
    for distribution in distributions:
        # Fits data to distribution and returns MLEs of scale and loc
        # pars -> mu = loc, sigma = scale
        pars = distribution.fit(data)
        # negative loglikelihood: -sum(log pdf(x, theta), axis=0)
        mle = distribution.nnlf(pars, data)
        mles.append(mle)

    results = [(distribution.name, mle)
               for distribution, mle in zip(distributions, mles)]
    # Sorts smallest to largest -- smallest NNL is best
    best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
    # print('Best fit reached using {}, MLE value: {}'.format(
    #     best_fit[0].name, best_fit[1]))

    # Return best fitting distribution and parameters (loc and scale)
    return [best_fit[0], best_fit[0].fit(data)]


def summarize(df):
    """
    Summarizes features across df by getting mean and stdev across each column.
    """
    class_summaries = {}
    # get distribution of each feature
    for column_name in df:
        if column_name != TARGET_LABEL:
            col_values = df[column_name].dropna(axis=0)
            if len(col_values) > 0:
                class_summaries[column_name] = find_best_fitting_dist(col_values)

    return class_summaries


def separate_classes(train):
    """
    Separate by class (of unique transient types)
    Return map of {transient type : DataFrame of samples of that type}
    """
    transient_classes = list(train[TARGET_LABEL].unique())
    separated_classes = {}
    priors = {}  # Prior value of class, based on frequency
    total_count = train.shape[0]
    for transient in transient_classes:
        sub_df = train.loc[train[TARGET_LABEL] == transient]

        # SET PRIOR value, by frequency of class in total set
        # Uniform prior
        priors[transient] = 1 / len(transient_classes)
        class_count = sub_df.shape[0]
        # Inverted Frequency-based prior
        # priors[transient] = 1 - (class_count / total_count)

        # Set class value
        sub_df.drop([TARGET_LABEL], axis=1, inplace=True)
        separated_classes[transient] = sub_df
    print("Unique transient types: " + str(len(separated_classes)))

    return separated_classes, priors


def summarize_by_class(training_dataset):
    separated, priors = separate_classes(training_dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries, priors
