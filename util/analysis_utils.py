import configparser
import csv
import json
import os
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import shap
import stellargraph
import lime
import torch
import torch_geometric
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from graphlime import GraphLIME
from stellargraph.layer import GraphConvolution

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

model_dir = config['model_dir']
analysis_dir = config['analysis_dir']
targets_dir = config['targets_dir']
aas_dir = config['aas_dir']

node_feature_names = ["Residue name",
                      "Residue number",
                      "B-factor",
                      "X coordinate",
                      "Y coordinate",
                      "Z coordinate"]

"""
https://medium.com/stellargraph/https-medium-com-stellargraph-saliency-maps-for-graph-machine-learning-5cca536974da
"""


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


# TODO: class aggregation
def class_aggregation(feature_ranks, dest_dir, mode):
    if mode not in ["positive", "negative", "all"]:
        raise ValueError("Class aggregation mode must be 'positive', 'negative' or 'all'.")

    filepath = os.path.join(dest_dir, f"feature_aggregation_{mode}.csv")
    feature_rank_analysis(feature_ranks, filepath)
    feature_correlations(filepath)


def feature_rank_analysis(feature_ranks, filepath):
    """
    Analyse and plot each feature's ranked importance.
    Make csv log of observed and calculated data.

    Args:
        feature_ranks: List of rank arrays, for each feature. Numbers represent the counts of rank (index+1)
        filepath: Path to csv log file

    """

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


def feature_correlations(log_path):
    """
    TODO: more detail
    Analyse correlations between important features. Are there features that tend to be important together?
    This might indicate that certain sets of features are more informative for the model.
    """
    df = pd.read_csv(log_path)
    correlation_matrix = df.corr()

    # Visualize the correlation matrix as a heatmap
    # TODO: move to vu and define plot paths; if useful
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlations')
    plt.savefig(f'./tmp/correlation.png')


def aa_freq_analysis(predictions, out_dir):
    with open(os.path.join(aas_dir, "aa_freqs_init.json"), "r") as json_file:
        aa_freq_data = json.load(json_file)

    with open(os.path.join(aas_dir, "aas_by_protein.json"), "r") as json_file:
        aas_by_protein = json.load(json_file)

    for label, prediction, true_class in predictions:
        for aa in aas_by_protein[label.strip(".pdb")]["unique_aas"]:
            if round(prediction[0]) == round(true_class):
                if round(true_class) == 1:
                    aa_freq_data[aa]["pred_pos_correct"] += 1
                else:
                    aa_freq_data[aa]["pred_neg_correct"] += 1
            else:
                if round(true_class) == 1:
                    aa_freq_data[aa]["pred_pos_incorrect"] += 1
                else:
                    aa_freq_data[aa]["pred_neg_incorrect"] += 1

    json_file_path = os.path.join(out_dir, "aa_freqs.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(aa_freq_data, json_file, indent=4)


def extract_popular_aas(run_dir, top_n=5):
    with open(os.path.join(run_dir, "amino_acids", "aa_freqs.json"), "r") as json_file:
        aa_freq_data = json.load(json_file)

    # Create lists to store data
    amino_acids = []
    popularity_ratios_pos = []
    popularity_ratios_neg = []
    correctness_percentages_pos = []
    correctness_percentages_neg = []

    # Calculate popularity ratios and correctness percentages
    for aa, data in aa_freq_data.items():
        # Calculate popularity ratio in prediction
        total_pos = int(data["pred_pos_correct"]) + int(data["pred_pos_incorrect"])
        total_neg = int(data["pred_neg_correct"]) + int(data["pred_neg_incorrect"])
        total_preds = total_pos + total_neg

        # skip the proteins inference wasn't done on
        if total_pos == 0 and total_neg == 0:
            continue
        else:
            amino_acids.append(aa)

        if total_pos > 0:
            pos_ratio = round(total_pos / total_preds, 2)
            correctness_percentage_pos = (data["pred_pos_correct"] / total_pos) * 100 if total_pos > 0 else 0

            popularity_ratios_pos.append(pos_ratio)
            correctness_percentages_pos.append(correctness_percentage_pos)

        if total_neg > 0:
            neg_ratio = round(total_neg / total_preds, 2)
            correctness_percentage_neg = (data["pred_neg_correct"] / total_neg) * 100 if total_neg > 0 else 0

            popularity_ratios_neg.append(neg_ratio)
            correctness_percentages_neg.append(correctness_percentage_neg)

    df_pos = pd.DataFrame({
        'amino_acid': amino_acids,
        'popularity_ratio': popularity_ratios_pos,
        'correctness_percentage': correctness_percentages_pos,
        'prediction_type': 'positive'
    })

    df_neg = pd.DataFrame({
        'amino_acid': amino_acids,
        'popularity_ratio': popularity_ratios_neg,
        'correctness_percentage': correctness_percentages_neg,
        'prediction_type': 'negative'
    })

    # extract top_n most popular ones
    df_pos.sort_values(by='popularity_ratio', ascending=False, inplace=True)
    df_neg.sort_values(by='popularity_ratio', ascending=False, inplace=True)

    # identify the value of popularity ratio for the Nth row
    threshold_pos = df_pos.iloc[top_n - 1]['popularity_ratio']
    threshold_neg = df_neg.iloc[top_n - 1]['popularity_ratio']

    # keep all rows with the same popularity ratio as the top_n-th row or higher
    df_pos = df_pos[df_pos['popularity_ratio'] >= threshold_pos]
    df_neg = df_neg[df_neg['popularity_ratio'] >= threshold_neg]

    # reset index for readibility
    df_pos.reset_index(drop=True, inplace=True)
    df_neg.reset_index(drop=True, inplace=True)

    df_result = pd.concat([df_pos, df_neg])

    df_result.reset_index(drop=True, inplace=True)

    output_csv_path = os.path.join(run_dir, "amino_acids", "popularity_rankings.csv")
    df_result.to_csv(output_csv_path, index=False)


# FIXME: debilno
# """
def f(X):
    print(f"f type: {type(X)}")
    # print(X)
    data = X[0][0]
    # print(data)
    print(f"data type: {type(data)}")
    # FIXME: posebno debilno
    with tf.keras.utils.custom_object_scope({'GraphConvolution': GraphConvolution}):
        model = tf.keras.models.load_model(os.path.join(model_dir, "gcn_model.h5"))

    return model.predict(data)  # [X[:, i] for i in range(X.shape[1])]).flatten()


def perform_shap(model, inference_tensors, plot_path):
    """
    SHAP values assign a value to each feature for a particular prediction, indicating its contribution to that prediction
    """

    # TODO: Will this make a new prediction that needn't be the same as the inference one?

    # tensor_array = inference_tensors.to_numpy()

    # graph extraction attempt 1
    # inference_nx = [stellargraph.StellarGraph.to_networkx(graph) for graph in inference_tensors]
    # inference_np = [nx.to_numpy_array(graph) for graph in inference_nx]
    # print(inference_np)

    # graph extraction attempt 2
    # same as extracting from nx
    graphs_from_generator = [np.squeeze(inference_tensors.__getitem__(idx)[0][0], axis=0) for idx in
                             range(3)]  # TODO: ne hardkodirati 3
    graph_shapes = [graph.shape for graph in graphs_from_generator]
    largest_shape = max(graph_shapes, key=lambda x: np.prod(x))

    padded_graphs = [
        np.pad(
            graph,
            [
                (0, largest_shape[0] - graph.shape[0]),
                (0, largest_shape[1] - graph.shape[1])
            ],
            mode='constant'
        )
        for graph in graphs_from_generator
    ]

    for padded_graph in padded_graphs:
        print(padded_graph.shape)

    # explainer definition attempt 1
    # explainer = shap.KernelExplainer(f, inference_tensors)
    # shap_values_bin = explainer(padded_graphs)
    # print(shap_values_bin)

    # explainer definition attempt 2
    explainer = shap.KernelExplainer(f, data=pd.Series([inference_tensors]))
    shap_values = explainer.shap_values(pd.Series(inference_tensors), nsamples=3)

    # getting relevant data
    # shap.force_plot(explainer.expected_value, shap_values, inference_tensors)
    plt.savefig(plot_path, dpi=700)
    print(shap_values)

    # needed?
    """return shap.force_plot(explainer.expected_value[1],
                           shap_values[1],
                           inference_tensors,
                           matplotlib=True,
                           show=False)"""


def get_torch_geometric(inference_nx):
    data_list = []
    for nx_graph in inference_nx:
        # node features
        node_features = torch.tensor(list(nx.get_node_attributes(nx_graph, "feature").values()), dtype=torch.float)

        # Create a mapping from node names to unique integer identifiers
        node_name_to_id = {node_name: idx for idx, node_name in enumerate(nx_graph.nodes())}

        # edges
        edges = np.array([(node_name_to_id[edge[0]], node_name_to_id[edge[1]]) for edge in nx_graph.edges()])
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        # edge_index = torch.tensor(list(nx_graph.edges()).T, dtype=torch.long)

        # Create PyTorch Geometric Data object
        data = torch_geometric.data.Data(x=node_features, edge_index=edge_index)

        # Add other attributes if needed, e.g., edge features (edge_attr), etc.

        # Append to the list
        data_list.append(data)

    return data_list


def perform_lime(model, inference_graphs, plot_path):
    # data = inference_tensors  # a `torch_geometric.data.Data` object
    model = model  # any GNN model
    node_idx = 0  # the specific node to be explained

    print(type(model))
    if isinstance(model, tf.keras.Model):
        tf.keras.backend.set_learning_phase(True)

    inference_nx = [stellargraph.StellarGraph.to_networkx(graph) for graph in inference_graphs]
    inference_np = [nx.to_numpy_array(graph) for graph in inference_nx]

    data_list = get_torch_geometric(inference_nx)
    data = data_list[0]  # Choose the appropriate data object

    # Pass numpy arrays to the explainer
    explainer = GraphLIME(model, hop=2, rho=0.1)
    coefs = explainer.explain_node(node_idx, data.x, data.edge_index)

    # get relevant data
    print(coefs)
    plt.figure(figsize=(16, 4))

    x = list(range(data.num_node_features))
    plt.bar(x, coefs, width=5.0)
    plt.xlabel('Feature Index')
    plt.ylabel(r'$\beta$');

    print(f'The {np.argmax(coefs)}-th feature is the most important.')


def generate_pdps():
    """
    TODO: Partial Dependence Plots (PDP)
    PDPs show the relationship between a feature and the model's predictions while keeping other features constant.
    This can reveal how a single feature impacts the model's output across its entire range.
    """
    pass
