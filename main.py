import configparser
import csv
import logging
import os
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from stellargraph.layer import GraphConvolution, SortPooling
from stellargraph.mapper import PaddedGraphGenerator

from model.model import in_out_tensors
from model.train import train_model
from util import file_utils as fu, graph_utils as gu, visualization_utils as vu, analysis_utils as au

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

check_setup = config['check_setup']
one_per_entry = config['one_per_entry']
demo_run = config['demo_run']
use_dgcnn = config['use_dgcnn']

targets_dir = config['targets_dir']
pdb_catalytic_dir = config['pdb_catalytic_dir']
pdb_non_catalytic_dir = config['pdb_non_catalytic_dir']
pdb_demo_dir = config['pdb_demo_dir']

graph_dir = config['graph_dir']
demo_graph_dir = config['demo_graph_dir']
model_dir = config['model_dir']
visualization_dir = config['visualization_dir']

analysis_dir = config['analysis_dir']
shap_dir = config['shap_dir']

# suppress "FutureWarning: The default value of regex will change from True to False in a future version." for graph
# generation
warnings.simplefilter(action='ignore', category=FutureWarning)


def check_and_generate_targets():
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        if demo_run.lower() == "y":
            fu.generate_targets([pdb_demo_dir])
        else:
            fu.generate_targets([pdb_catalytic_dir, pdb_non_catalytic_dir])
        logging.info("Finished target generation")


def generate_graphs():
    if demo_run.lower() == "y":
        if not os.listdir(demo_graph_dir):
            [gu.generate_graph(pdb_demo_dir, entry.replace(".pdb", ""), demo_graph_dir) for entry in
             os.listdir(pdb_demo_dir)]
            logging.info("Generated demo graphs")

        # TODO: add tagret generation

    else:
        if not os.listdir(graph_dir):
            df = pd.read_csv(os.path.join(targets_dir, "PDBannot.txt"), sep="\s+", skip_blank_lines=True,
                             na_values=[''])
            df[['Residue_1', 'Residue_2', 'Residue_3']] = df[['Residue_1', 'Residue_2', 'Residue_3']].fillna(-1)
            df[['Residue_1', 'Residue_2', 'Residue_3']] = df[['Residue_1', 'Residue_2', 'Residue_3']].astype(int)

            # Delete targets and start anew every time
            targets_file_path = os.path.join(targets_dir, "targets.txt")
            if os.path.exists(targets_file_path):
                os.remove(targets_file_path)

            # Generate graphs only for entries in pdb_catalytic_dir that have corresponding entries in df["PDB_ID"]
            for entry in os.listdir(pdb_catalytic_dir):
                pdb_id = entry.replace(".pdb", "")
                if pdb_id in df["PDB_ID"].values:
                    if gu.generate_graph(pdb_catalytic_dir, pdb_id, graph_dir):
                        with open(targets_file_path, "a") as targets_file:
                            targets_file.write(f"{pdb_id}\t1\n")
                    else:
                        logging.warning(f"Failed to generate graph for {pdb_id}")
            logging.info("Generated catalytic graphs")

            # Generate graphs for non-catalytic entries by randomly picking in the same quantity as catalytic entries
            non_catalytic_entries = random.sample(os.listdir(pdb_non_catalytic_dir), len(os.listdir(graph_dir)))
            for entry in non_catalytic_entries:
                pdb_id = entry.replace(".pdb", "")
                if gu.generate_graph(pdb_non_catalytic_dir, pdb_id, graph_dir):
                    with open(targets_file_path, "a") as targets_file:
                        targets_file.write(f"{pdb_id}\t0\n")
                else:
                    logging.warning(f"Failed to generate graph for {pdb_id}")
            logging.info("Generated non-catalytic graphs")


def load_graphs_and_labels():
    if demo_run.lower() == "y":
        return gu.load_graphs(demo_graph_dir), gu.load_graph_labels()
    else:
        return gu.load_graphs(graph_dir), gu.load_graph_labels()


def load_model():
    if use_dgcnn.lower() == "y":
        if not os.path.exists(os.path.join(model_dir, "dgcnn_model.h5")):
            return None

        with tf.keras.utils.custom_object_scope({'SortPooling': SortPooling, 'GraphConvolution': GraphConvolution}):
            model = tf.keras.models.load_model(os.path.join(model_dir, "dgcnn_model.h5"))
            print(model.summary())
    else:
        if not os.path.exists(os.path.join(model_dir, "gcn_model.h5")):
            return None

        with tf.keras.utils.custom_object_scope({'GraphConvolution': GraphConvolution}):
            model = tf.keras.models.load_model(os.path.join(model_dir, "gcn_model.h5"))
            print(model.summary())

    return model


def perform_model_training(generator, labels):
    model = None
    if use_dgcnn.lower() == "y":
        if "dgcnn_model.h5" not in os.listdir(model_dir):
            # Create and train classification models
            model = train_model(generator, labels, epochs=200, folds=10, n_repeats=5)
            print(model.summary())

            # Save the model
            model.save(os.path.join(model_dir, "dgcnn_model.h5"))
            print("GCN model trained and saved successfully.")

    else:
        if "gcn_model.h5" not in os.listdir(model_dir):
            # Create and train classification models
            model = train_model(generator, labels, epochs=200, folds=10, n_repeats=5)
            print(model.summary())

            # Save the model
            model.save(os.path.join(model_dir, "gcn_model.h5"))
            print("GCN model trained and saved successfully.")

    return model


if __name__ == "__main__":
    fu.check_setup(check_setup)

    if one_per_entry.lower() == "y":
        fu.pick_best()

    check_and_generate_targets()

    # Load or generate graphs
    generate_graphs()
    
    # Prepare input graph data for training and testing
    graphs, graph_labels = load_graphs_and_labels()
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, graph_labels, test_size=0.2, random_state=42
    )
    graph_generator = PaddedGraphGenerator(graphs=train_graphs)
    inference_generator = PaddedGraphGenerator(graphs=test_graphs)
    training_tensors = graph_generator.flow(train_graphs, weighted=True, targets=train_labels)
    inference_tensors = inference_generator.flow(test_graphs, weighted=True, targets=test_labels)

    # Load or train model
    if not load_model():
        model = perform_model_training(graph_generator, train_labels)
    else:
        model = load_model()
        if model is None:
            raise ValueError("Model cannot be None")

    # Initialise visualisation/run directory
    fu.create_folder(visualization_dir)

    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.mkdir(os.path.join(visualization_dir, f"{run_timestamp}"))
    run_dir = os.path.join(os.path.join(visualization_dir, f"{run_timestamp}"))

    # Initialise analysis directories
    fu.create_folder(analysis_dir)
    analysis_run_dir = os.path.join(os.path.join(analysis_dir, f"{run_timestamp}"))
    os.mkdir(analysis_run_dir)

    os.mkdir(os.path.join(analysis_run_dir, "amino_acids"))

    # Make predictions using the loaded model
    predictions = model.predict(inference_tensors)

    # Visualise predictions histogram
    # TODO: uncomment when fixed & pass testing graphs as obj
    # vu.visualise_predictions(predictions, test_labels.to_list(), os.path.join(run_dir, "predictions"))

    # Convert the predictions to binary class labels (0 or 1)
    binary_predictions = np.round(predictions).astype(int)

    x_t, mask, A_m = in_out_tensors(inference_generator, model)[0]
    inputs = [x_t, mask, A_m]

    # Compute and visualise Grad-CAM heatmaps for each sample in the inference dataset
    features_ranked_all = [[0 for j in range(inference_generator.node_features_size)] for i in
                           range(inference_generator.node_features_size)]

    features_ranked_positive = [[0 for j in range(inference_generator.node_features_size)] for i in
                                range(inference_generator.node_features_size)]

    features_ranked_negative = [[0 for j in range(inference_generator.node_features_size)] for i in
                                range(inference_generator.node_features_size)]

    fu.generate_aa_frequencies()
    ranks_log_filepath = os.path.join(analysis_run_dir, f"feature_ranks.csv")

    most_relevant_nodes = {}  # dict of protein: [nodes with the largest gradients]
    for i, graph in enumerate(test_graphs):
        prediction = binary_predictions[i][0]
        print(f"Graph {i + 1} - {test_labels.index[i]}:\n"
              f"Predicted class - {prediction} ({predictions[i][0]:.2f})\n\t True class - {round(test_labels[i])}\n")

        # Get the input features for the sample
        inputs = inference_tensors[i][0]
        if inputs is None:
            print(f"Skipping graph {i + 1} due to None input.")
            continue

        # Gradients - extract and append to list for further analysis
        gradients = vu.get_gradients(model, inputs)
        node_gradients = gradients[0]
        edge_gradients = gradients[-1]

        most_relevant_nodes[test_labels.index[i]] = au.extract_relevant_gradients(node_gradients)

        # Saliency maps
        node_saliency_map = vu.calculate_node_saliency(gradients[0])
        edge_saliency_map = vu.calculate_edge_saliency(gradients[-1])

        # Feature importance ranking
        feature_importance = np.mean(np.abs(node_gradients[0].numpy()), axis=0)
        feature_ranking = np.argsort(feature_importance)[::-1]

        with open(ranks_log_filepath, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            if os.path.getsize(ranks_log_filepath) == 0:
                writer.writerow(
                    ["Residue name", "Residue number", "B-factor", "X coordinate", "Y coordinate", "Z coordinate"])
            writer.writerow(list(feature_ranking))

        # Print feature importance ranking
        features_ranked = []
        for rank, feature_index in enumerate(feature_ranking):
            if test_labels[i] == 1:
                features_ranked_positive[feature_index][rank - 1] += 1
            else:
                features_ranked_negative[feature_index][rank - 1] += 1
            features_ranked_all[feature_index][rank - 1] += 1

        # Visualize the saliency maps and save them as images
        vu.visualize_node_heatmap(node_saliency_map, os.path.join(run_dir, f"node_saliency_map-{i}.png"))
        vu.visualize_edge_heatmap(edge_saliency_map, os.path.join(run_dir, f"edge_saliency_map-{i}.png"))

    # Active site comparison (gradient vs. ground truth)
    au.active_site_comparison(most_relevant_nodes, analysis_run_dir)
    au.generate_triad_combinations(most_relevant_nodes, analysis_run_dir)

    # class-aggregated analysis
    au.class_aggregation(features_ranked_all, analysis_run_dir, "all")
    au.class_aggregation(features_ranked_positive, analysis_run_dir, "positive")
    au.class_aggregation(features_ranked_negative, analysis_run_dir, "negative")

    # Correlation matrix of feature ranking in inference
    vu.feature_correlations(ranks_log_filepath, analysis_run_dir)

    vu.evaluate_model(binary_predictions, test_labels, both_classes_present=False)
    vu.save_feature_rankings(features_ranked_all, os.path.join(run_dir, "feature_rankings.txt"))
