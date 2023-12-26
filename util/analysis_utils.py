import configparser
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

import util.file_utils as fu

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
    # Rank that occurred most times, so index in array with most occurrences (+1)
    feature_max = max(data)
    indices = [str(idx + 1) for idx, value in enumerate(data) if value == feature_max]
    return ';'.join(indices)


def calc_least_frequent(data):
    # Rank that occurred the least times, so index in array with most occurrences (+1)
    feature_min = min(data)
    indices = [str(idx + 1) for idx, value in enumerate(data) if value == feature_min]
    return ';'.join(indices)


def class_aggregation(feature_ranks, dest_dir, mode):
    if mode not in ["positive", "negative", "all"]:
        raise ValueError("Class aggregation mode must be 'positive', 'negative' or 'all'.")

    feature_dir = os.path.join(dest_dir, "features")
    fu.create_folder(feature_dir)

    feature_rank_analysis(feature_ranks, feature_dir, mode)


def feature_rank_analysis(feature_ranks, dest_dir, run_mode):
    """
    Analyse and plot each feature's ranked importance.
    Make csv log of observed and calculated data.

    Args:
        feature_ranks: List of rank arrays, for each feature. Numbers represent the counts of rank (index+1)
        dest_dir: Path to csv log file

    """

    for feature, rank_occurrences in zip(node_feature_names, feature_ranks):
        feature_aggregation_data = []

        mean = calc_mean(rank_occurrences)
        median_rank = calc_median_idx(rank_occurrences)
        stdev = calc_stdev(rank_occurrences, mean)

        mode = calc_mode(rank_occurrences)
        least_frequent = calc_least_frequent(rank_occurrences)

        feature_aggregation_data.append([feature, mode, least_frequent, mean, median_rank, stdev])

        plot_feature_analysis(feature, rank_occurrences, mean, median_rank, dest_dir)

        filepath = os.path.join(dest_dir, f"feature_aggregation_{run_mode}.csv")
        with open(filepath, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if os.path.getsize(filepath) == 0:
                writer.writerow(['feature', 'mode', 'least_frequent', 'mean', 'median', 'stdev'])
            writer.writerows(feature_aggregation_data)


def plot_feature_analysis(feature, rank_occurrences, mean, median, dest_dir):
    rank_values = [rank + 1 for rank in range(len(rank_occurrences))]

    plt.clf()
    plt.bar(rank_values, rank_occurrences)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'{feature} rank distribution')

    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label='Median')

    plt.legend()
    filepath = os.path.join(dest_dir, f"{feature}.png")
    plt.savefig(filepath)


def aa_freq_analysis(predictions, out_dir):
    with open(os.path.join(aas_dir, "aa_freqs_init.json"), "r") as json_file:
        aa_freq_data = json.load(json_file)

    with open(os.path.join(aas_dir, "aas_by_protein.json"), "r") as json_file:
        aas_by_protein = json.load(json_file)

    for label, prediction, true_class in predictions:
        if label.strip(".pdb") in aas_by_protein:
            for aa in aas_by_protein[label.strip(".pdb")]["unique_aas"]:
                if round(prediction[0]) == round(true_class):
                    if 0.6 <= true_class <= 1:  # this must always match the set threshold
                        aa_freq_data[aa]["true_positives"] += 1
                    else:
                        aa_freq_data[aa]["true_negatives"] += 1
                else:
                    if 0.6 <= true_class <= 1:
                        aa_freq_data[aa]["false_positives"] += 1
                    else:
                        aa_freq_data[aa]["false_negatives"] += 1
        else:
            print(f"Skipping {label} due to misformated input")  # FIXME: should not be happening?

    json_file_path = os.path.join(out_dir, "aa_freqs.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(aa_freq_data, json_file, indent=4)


def extract_popular_aas(run_dir, top_n=5):
    with open(os.path.join(run_dir, "amino_acids", "aa_freqs.json"), "r") as json_file:
        aa_freq_data = json.load(json_file)

    # Create lists to store data
    amino_acids_pos = []
    precisions = []
    recalls = []

    # Calculate popularity measures for each AA
    for aa, data in aa_freq_data.items():
        # Inspect precision and recall for positive class, if present
        if (data['true_positives'] + data['false_negatives']) > 0:
            precision = round(data['true_positives'] / (data['true_positives'] + data['false_positives']), 2)
            recall = round(data["true_positives"] / (data['true_positives'] + data['false_negatives']), 2)

            amino_acids_pos.append(aa)
            precisions.append(precision)
            recalls.append(recall)

    df = pd.DataFrame({
        'amino_acid': amino_acids_pos,
        'precision': precisions,
        'recall': recalls
    })

    df.sort_values(by='recall', ascending=False, inplace=True)
    # TODO: look into other metrics

    # extract top_n most popular ones
    # threshold = df_pos.iloc[top_n - 1]['recall']
    # df_pos = df_pos[df_pos['recall'] >= threshold]

    output_csv_path = os.path.join(run_dir, "amino_acids", "pos_rankings.csv")
    df.to_csv(output_csv_path, index=False)


def extract_relevant_gradients(gradients):
    # Min-max normalisation for each tensor
    min_values = [tf.reduce_min(tensor) for tensor in gradients]
    max_values = [tf.reduce_max(tensor) for tensor in gradients]

    normalised_tensors = [(tensor - min_val) / (max_val - min_val) for tensor, min_val, max_val in
                          zip(gradients, min_values, max_values)]

    all_gradients = pd.DataFrame(np.concatenate(normalised_tensors, axis=0))  # reshaping

    # Calculate the threshold for each feature independently
    # TODO: how to define threshold?
    threshold_percentage = 98
    thresholds = np.percentile(all_gradients, threshold_percentage, axis=0)
    thresholds = thresholds.tolist()

    # Extract relevant gradients for each feature
    filtered_nodes = []
    for col_idx, threshold in enumerate(thresholds):
        column = all_gradients[col_idx]
        filtered_nodes.append(column[column > threshold].index.tolist())

    flat_list = [item for sublist in filtered_nodes for item in sublist]
    unique_nodes = list(set(flat_list))

    return unique_nodes


def active_site_comparison(data):
    """
    Args:
        data: dict - a dictionary containing all relevant nodes per protein
    Returns:

    """
    df = pd.read_csv(os.path.join(targets_dir, "PDBannot.txt"), sep="\s+", skip_blank_lines=True, na_values=[''])
    df[['Residue_1', 'Residue_2', 'Residue_3']] = df[['Residue_1', 'Residue_2', 'Residue_3']].fillna(-1)
    df[['Residue_1', 'Residue_2', 'Residue_3']] = df[['Residue_1', 'Residue_2', 'Residue_3']].astype(int)

    inference_proteins = data.keys()
    results = []
    for index, row in df.iterrows():
        protein = row['PDB_ID']
        if protein not in inference_proteins:
            print(f"{protein} not in inference. Skipping.")
        else:
            nodes_list = [node + 1 for node in data[protein]]
            res1 = int(row['Residue_1'])
            res2 = int(row['Residue_2'])
            res3 = int(row['Residue_3'])

            # Check if each residue is present in the nodes list
            res1_present = res1 in nodes_list
            res2_present = res2 in nodes_list
            res3_present = res3 in nodes_list

            # Check if all three residues are present
            all_present = all([res1_present, res2_present, res3_present])

            # Append the results to the list
            results.append({
                'protein': protein,
                'res1_present': res1_present,
                'res2_present': res2_present,
                'res3_present': res3_present,
                'all_res_present': all_present
            })

    results_df = pd.DataFrame(results)
    print(results_df)

    # TODO: make triad combinations and compare that
