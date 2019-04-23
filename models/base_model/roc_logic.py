"""
Logic for ROC curves, shared between BaseModelPerformance and MCBaseModelPerformance
"""
import numpy as np


def get_normal_pdf(probabilities):
    """
    Returns normal PDF values
    """
    samples = np.array(probabilities)
    mean = np.mean(samples)
    std = np.sqrt(np.var(samples))
    x = np.linspace(0, 1, num=100)
    # Fit normal distribution to mean and std of data
    if std == 0:
        const = 0
    else:
        const = 1.0 / np.sqrt(2 * np.pi * (std**2))
    y = const * np.exp(-((x - mean)**2) / (2.0 * (std**2)))

    return x, y


def get_fp_tp_rates(x, pos_pdf, neg_pdf):
    # Sum of all probabilities
    total_class = np.sum(pos_pdf)
    total_not_class = np.sum(neg_pdf)

    area_TP = 0  # Total area
    area_FP = 0  # Total area under incorrect curve

    TP_rates = []  # True positive rates
    FP_rates = []  # False positive rates
    # For each data point in x
    for i in range(len(x)):
        if pos_pdf[i] > 0:
            area_TP += pos_pdf[len(x) - 1 - i]
            area_FP += neg_pdf[len(x) - 1 - i]
        # Calculate FPR and TPR for threshold x
        # Volume of false positives over total negatives
        FPR = area_FP / total_not_class
        # Volume of true positives over total positives
        TPR = area_TP / total_class
        TP_rates.append(TPR)
        FP_rates.append(FPR)

    # Plotting final ROC curve, FP against TP
    return FP_rates, TP_rates
