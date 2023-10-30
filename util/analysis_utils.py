import configparser
import os

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

node_feature_names = ["Residue name",
                      "Residue number",
                      "B-factor",
                      "X coordinate",
                      "Y coordinate",
                      "Z coordinate"]


def calc_weighted_mean(data):
    weighted_ranks = [data[idx] * (idx + 1) for idx in range(len(data))]
    weighted_sum = sum(weighted_ranks)
    total_count = sum(data)
    return weighted_sum / total_count


def calc_weighted_median_idx(data):
    cumulative_freq = np.cumsum(data)
    total_count = sum(data)

    median_rank = total_count / 2
    median_index = np.where(cumulative_freq >= median_rank)[0][0]

    return median_index + 1


def calc_weighted_stdev(data, mean):
    total_count = sum(data)
    weighted_rank_values = [rank for rank, freq in enumerate(data, 1) for _ in range(freq)]

    weighted_variance = sum(
        [(val - mean) ** 2 * freq for val, freq in zip(weighted_rank_values, data)]) / total_count

    return np.sqrt(weighted_variance)


"""
TODO: Aggregation
Group proteins by class and compute summary statistics (mean, median, etc.) for feature importance scores within each class.
This can help you identify class-specific patterns.
"""


def aggregation(feature_ranks):
    # TODO: add type-based csv log and visualisation destination

    for feature, row in zip(node_feature_names, feature_ranks):
        weighted_mean = calc_weighted_mean(row)

        weighted_median_rank = calc_weighted_median_idx(row)

        weighted_std_deviation = calc_weighted_stdev(row, weighted_mean)

        # TODO: move to VU - Create a weighted histogram
        # FIXME: plotting
        rank_values = [rank for rank, freq in enumerate(row, 1) for _ in range(freq)]

        plt.clf()
        plt.hist(rank_values, bins=len(row), weights=rank_values, edgecolor='k', alpha=0.75)
        plt.xlabel('Rank')
        plt.ylabel('Weighted Frequency')
        plt.title('Weighted Rank Distribution')

        # Add weighted mean and median to the plot
        plt.axvline(weighted_mean, color='red', linestyle='dashed', linewidth=2, label='Weighted Mean')
        plt.axvline(weighted_median_rank, color='green', linestyle='dashed', linewidth=2, label='Weighted Median')

        plt.legend()
        plt.savefig(f'./tmp/{feature}.png')

        print(f"Mean Rank for {feature}: {weighted_mean}")
        print(f"Median Rank for {feature}: {weighted_median_rank}")
        print(f"Standard Deviation of Ranks for {feature}: {weighted_std_deviation}\n")


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
