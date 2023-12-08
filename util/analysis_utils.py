import configparser
import csv
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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
    # Rank that occurred most times, so index in array with most occurences (+1)
    feature_max = max(data)
    indices = [str(idx + 1) for idx, value in enumerate(data) if value == feature_max]
    return ';'.join(indices)


def calc_least_frequent(data):
    # Rank that occurred least times, so index in array with most occurences (+1)
    feature_min = min(data)
    indices = [str(idx + 1) for idx, value in enumerate(data) if value == feature_min]
    return ';'.join(indices)


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
    # move to vu and define plot paths; if useful
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
        if label.strip(".pdb") in aas_by_protein:
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
        else:
            print(f"Skipping {label} due to misformated input")  # FIXME: should not be happening

    json_file_path = os.path.join(out_dir, "aa_freqs.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(aa_freq_data, json_file, indent=4)


def extract_popular_aas(run_dir, top_n=5):
    with open(os.path.join(run_dir, "amino_acids", "aa_freqs.json"), "r") as json_file:
        aa_freq_data = json.load(json_file)

    # Create lists to store data
    amino_acids_pos = []
    amino_acids_neg = []
    popularity_ratios_pos = []
    popularity_ratios_neg = []
    correctness_percentages_pos = []
    correctness_percentages_neg = []

    # Calculate popularity ratios and correctness percentages
    for aa, data in aa_freq_data.items():
        # Calculate popularity ratio in prediction
        tp = float(data["pred_pos_correct"])
        tn = float(data["pred_neg_correct"])
        fn = float(data["pred_neg_incorrect"])
        fp = float(data["pred_pos_incorrect"])
        pred_pos = tp + fn
        pred_neg = tn + fp
        total_preds = pred_pos + pred_neg

        # skip the proteins inference wasn't done on
        if pred_pos == 0 and pred_neg == 0:
            continue

        if pred_pos > 0:
            expected_ratio = pred_pos / total_preds  # expected ratio in a balanced dataset
            pos_ratio = tp / total_preds  # observed ratio

            normalised_pos_ratio = round(pos_ratio / expected_ratio, 2)
            correctness_percentage_pos = round((data["pred_pos_correct"] / pred_pos) * 100, 2) if pred_pos > 0 else 0

            amino_acids_pos.append(aa)
            popularity_ratios_pos.append(normalised_pos_ratio)
            correctness_percentages_pos.append(correctness_percentage_pos)

        if pred_neg > 0:
            expected_ratio = pred_neg / total_preds
            neg_ratio = tn / total_preds

            normalised_neg_ratio = round(neg_ratio / expected_ratio, 2)
            correctness_percentage_neg = round((data["pred_neg_correct"] / pred_neg) * 100, 2) if pred_neg > 0 else 0

            amino_acids_neg.append(aa)
            popularity_ratios_neg.append(normalised_neg_ratio)
            correctness_percentages_neg.append(correctness_percentage_neg)

    df_pos = pd.DataFrame({
        'amino_acid': amino_acids_pos,
        'popularity_ratio': popularity_ratios_pos,
        'correctness_percentage': correctness_percentages_pos,
        'prediction_type': 'positive'
    })

    df_neg = pd.DataFrame({
        'amino_acid': amino_acids_neg,
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
