import configparser
import csv
import os
import statistics

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
import tensorflow as tf
from stellargraph.layer import GraphConvolution

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

model_dir = config['model_dir']
analysis_dir = config['analysis_dir']

node_feature_names = ["Residue name",
                      "Residue number",
                      "B-factor",
                      "X coordinate",
                      "Y coordinate",
                      "Z coordinate"]


def calc_mean(data):
    ranks = [data[idx] * (idx + 1) for idx in range(len(data))]
    ranks_sum = sum(ranks)
    total_count = sum(data)
    return round(ranks_sum / total_count, 2)


def calc_median_idx(data):
    cumulative_freq = np.cumsum(data)
    total = sum(data)

    median_rank = total / 2
    median_index = np.where(cumulative_freq >= median_rank)[0][0]

    return median_index + 1


def calc_stdev(data, mean):
    total_count = sum(data)
    rank_values = [rank for rank, freq in enumerate(data, 1) for _ in range(freq)]

    variance = sum(
        [(val - mean) ** 2 * freq for val, freq in zip(rank_values, data)]) / total_count

    return round(np.sqrt(variance), 2)


def calc_mode(data):
    # Rank that occurred most times, so index in array with most occurences (+1)
    feature_max = max(data)
    indices = [str(idx + 1) for idx, value in enumerate(data) if value == feature_max]
    return ';'.join(indices)


def calc_least_frequent(data):
    # Rank that occurred least times, so index in array with most occurences (+1)
    feature_min = min(data)
    indices = [str(idx + 1) for idx, value in enumerate(data) if value == feature_min]
    return ';'.join(indices)


"""
TODO: Aggregation
Group proteins by class and compute summary statistics (mean, median, etc.) for feature importance scores within each class.
This can help you identify class-specific patterns.
"""


# TODO: class aggregation


def feature_aggregation(feature_ranks, dest_dir):
    filepath = os.path.join(dest_dir, "feature_aggregation.csv")

    for feature, rank_occurences in zip(node_feature_names, feature_ranks):
        feature_aggregation_data = []

        mean = calc_mean(rank_occurences)
        median_rank = calc_median_idx(rank_occurences)
        stdev = calc_stdev(rank_occurences, mean)

        mode = calc_mode(rank_occurences)
        least_frequent = calc_least_frequent(rank_occurences)

        feature_aggregation_data.append([feature, mode, least_frequent, mean, median_rank, stdev])

        # TODO: move to VU - Create a bar with extras; add destination
        rank_values = [rank + 1 for rank in range(len(rank_occurences))]

        plt.clf()
        plt.bar(rank_values, rank_occurences)
        plt.xlabel('Rank')
        plt.ylabel('Frequency')
        plt.title(f'{feature} rank distribution')

        plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
        plt.axvline(median_rank, color='green', linestyle='dashed', linewidth=2, label='Median')

        plt.legend()
        plt.savefig(f'./tmp/{feature}.png')

        with open(filepath, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if os.path.getsize(filepath) == 0:
                writer.writerow(['feature', 'mode', 'least_frequent', 'mean', 'median', 'stdev'])
            writer.writerows(feature_aggregation_data)


"""
TODO: Feature Correlations:
Analyze correlations between important features. Are there features that tend to be important together?
This might indicate that certain sets of features are more informative for the model.
"""


def feature_correlations():
    pass


"""
TODO: SHAP (SHapley Additive exPlanations)
SHAP values assign a value to each feature for a particular prediction, indicating its contribution to that prediction.
By applying SHAP values to your model's predictions, you can understand how each feature affects the output. 
ou can then visualize these values or aggregate them to identify consistent patterns.
"""


# FIXME: debilno
# """
def f(X):
    # print(f"f type: {type(X)}")
    # print(X)
    data = X[0][0]
    # print(data)
    # FIXME: posebno debilno
    with tf.keras.utils.custom_object_scope({'GraphConvolution': GraphConvolution}):
        model = tf.keras.models.load_model(os.path.join(model_dir, "gcn_model.h5"))

    return model.predict(data)  # predict([X[:, i] for i in range(X.shape[1])]).flatten()


def perform_shap(model, inference_tensors, plot_path):
    # TODO: Will this make a new prediction that needn't be the same as the inference one?

    # TODO: don't pass whole inference tensors
    # print(f"perform shap type: {type(inference_tensors)}")
    # print(pd.Series(inference_tensors), type(pd.Series(inference_tensors)))
    # print(pd.Series(inference_tensors).values, type(pd.Series(inference_tensors).values))
    explainer = shap.KernelExplainer(f, data=pd.Series([inference_tensors]))
    shap_values = explainer.shap_values(pd.Series(inference_tensors), nsamples=100)
    shap.force_plot(explainer.expected_value, shap_values, inference_tensors)

    plt.savefig(plot_path, dpi=700)

    # needed?
    return shap.force_plot(explainer.expected_value[1],
                           shap_values[1],
                           inference_tensors,
                           matplotlib=True,
                           show=False)


"""


def f(data):
    # Load your model and perform predictions here
    with tf.keras.utils.custom_object_scope({'GraphConvolution': GraphConvolution}):
        model = tf.keras.models.load_model(os.path.join(model_dir, "gcn_model.h5"))

    return model.predict(data)


def perform_shap(model, inference_data, plot_path):
    explainer = shap.KernelExplainer(f, data=inference_data)  # Pass raw data, no need to create a Pandas Series
    shap_values = explainer.shap_values(inference_data, nsamples=100)
    shap.force_plot(explainer.expected_value, shap_values, inference_data)

    plt.savefig(plot_path, dpi=700)
    
    """

"""
TODO: Partial Dependence Plots (PDP)
PDPs show the relationship between a feature and the model's predictions while keeping other features constant.
This can reveal how a single feature impacts the model's output across its entire range.
"""


def generate_pdps():
    pass
