import numpy as np
import pandas as pd
import tensorflow as tf
from stellargraph.layer import GraphConvolution, SortPooling
from stellargraph.mapper import PaddedGraphGenerator

from model.train import train_model
from model.model import in_out_tensors
from util import file_utils as fu, graph_utils as gu, visualization_utils as vu, analysis_utils as au
from datetime import datetime

import os
import configparser
import logging
import warnings

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
pdb_inference_dir = config['pdb_inference_dir']
pdb_demo_dir = config['pdb_demo_dir']

graph_dir = config['graph_dir']
demo_graph_dir = config['demo_graph_dir']
inference_dir = config['inference_dir']
model_dir = config['model_dir']
categories_dir = config['categories_dir']
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


def generate_graphs_and_categories():
    if demo_run.lower() == "y":
        if not os.listdir(demo_graph_dir):
            [gu.generate_graph(pdb_demo_dir, entry.replace(".pdb", ""), demo_graph_dir) for entry in
             os.listdir(pdb_demo_dir)]
            logging.info("Generated demo graphs")

            # Generate graph categories
            gu.generate_categories(demo_graph_dir, categories_dir)
            logging.info("Generated graph categories")

    else:
        if not os.listdir(graph_dir):
            [gu.generate_graph(pdb_catalytic_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_catalytic_dir)]
            logging.info("Generated catalytic graphs")

            [gu.generate_graph(pdb_non_catalytic_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_non_catalytic_dir)]
            logging.info("Generated non-catalytic graphs")

            gu.generate_categories(graph_dir, categories_dir)
            logging.info("Generated graph categories")


def load_graphs_and_labels():
    if demo_run.lower() == "y":
        return gu.load_graphs(demo_graph_dir), gu.load_graph_labels()
    else:
        return gu.load_graphs(graph_dir), gu.load_graph_labels()


def generate_inference_graphs():
    if not os.listdir(inference_dir):
        [gu.generate_graph(pdb_inference_dir, entry.replace(".pdb", ""), inference_dir) for entry in
         os.listdir(pdb_inference_dir)]
        logging.info("Generated inference graphs")


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


def perform_model_training():
    model = None
    if use_dgcnn.lower() == "y":
        if "dgcnn_model.h5" not in os.listdir(model_dir):
            # Create and train classification models
            model = train_model(graph_generator, graph_labels, epochs=200, folds=10, n_repeats=5)
            print(model.summary())

            # Save the model
            model.save(os.path.join(model_dir, "dgcnn_model.h5"))
            print("GCN model trained and saved successfully.")

    else:
        if "gcn_model.h5" not in os.listdir(model_dir):
            # Create and train classification models
            model = train_model(graph_generator, graph_labels, epochs=200, folds=10, n_repeats=5)
            print(model.summary())

            # Save the model
            model.save(os.path.join(model_dir, "gcn_model.h5"))
            print("GCN model trained and saved successfully.")

    return model


if __name__ == "__main__":
    fu.check_setup(check_setup)

    if one_per_entry.lower() == "y":  # TODO: match check_setup and pick_best calls AAAAAAAAAAAAA
        fu.pick_best()

    check_and_generate_targets()

    graphs = []
    inference_graphs = []

    # Load or generate graphs
    if not load_model():
        # Create graphs for model
        generate_graphs_and_categories()
        graphs, graph_labels = load_graphs_and_labels()
        graph_generator = PaddedGraphGenerator(graphs=graphs)

        # Train model
        model = perform_model_training()
    else:
        fu.create_folder(categories_dir)
        if not os.listdir(categories_dir):
            gu.generate_categories(demo_graph_dir,
                                   categories_dir) if demo_run.lower() == "y" else gu.generate_categories(
                graph_dir, categories_dir)
            logging.info("Generated graph categories")

        model = load_model()

        if model is None:
            raise ValueError("Model cannot be None")

        # graphs, graph_labels = load_graphs_and_labels()

    # Generate and use inference graphs
    fu.create_folder(inference_dir)
    if not os.listdir(inference_dir):
        generate_inference_graphs()
    fu.generate_ground_truth(pdb_inference_dir)
    gu.generate_categories(inference_dir, categories_dir)
    inference_graphs = gu.load_graphs(inference_dir)
    inference_labels = gu.load_graph_labels("inference_truth.txt")

    # Prepare input graph data for inference
    inference_generator = PaddedGraphGenerator(graphs=inference_graphs)
    inference_tensors = inference_generator.flow(inference_graphs, weighted=True, targets=inference_labels)

    # Initialise visualisation/run directory
    fu.create_folder(visualization_dir)

    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.mkdir(os.path.join(visualization_dir, f"{run_timestamp}"))
    run_dir = os.path.join(os.path.join(visualization_dir, f"{run_timestamp}"))

    # Initialise analysis directories
    # TODO: move to routine if too many
    fu.create_folder(analysis_dir)
    analysis_run_dir = os.path.join(os.path.join(analysis_dir, f"{run_timestamp}"))
    os.mkdir(analysis_run_dir)

    os.mkdir(os.path.join(analysis_run_dir, "amino_acids"))

    # Make predictions using the loaded model
    predictions = model.predict(inference_tensors)

    # Visualise predictions histogram
    vu.visualise_predictions(predictions, inference_labels.to_list(), os.path.join(run_dir, "predictions"))

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

    fu.generate_aa_frequencies()  # TODO: if not already done

    for i, graph in enumerate(inference_graphs):
        prediction = binary_predictions[i][0]
        print(f"Graph {i + 1} - {inference_labels.index[i]}:\n"
              f"Predicted class - {prediction} ({predictions[i][0]:.2f})\n\t True class - {round(inference_labels[i])}")

        # Get the input features for the sample
        inputs = inference_tensors[i][0]
        if inputs is None:
            print(f"Skipping graph {i + 1} due to None input.")
            continue

        # Gradients
        gradients = vu.get_gradients(model, inputs)
        node_gradients = gradients[0]
        edge_gradients = gradients[-1]

        # Saliency maps
        node_saliency_map = vu.calculate_node_saliency(gradients[0])
        edge_saliency_map = vu.calculate_edge_saliency(gradients[-1])

        # Feature importance ranking
        feature_importance = np.mean(np.abs(node_gradients[0].numpy()), axis=0)
        feature_ranking = np.argsort(feature_importance)[::-1]

        # Print feature importance ranking
        features_ranked = []
        for rank, feature_index in enumerate(feature_ranking):
            if inference_labels[i] == 1:
                features_ranked_positive[feature_index][rank - 1] += 1
            else:
                features_ranked_negative[feature_index][rank - 1] += 1
            features_ranked_all[feature_index][rank - 1] += 1
            print(f"Rank {rank + 1}: Feature {feature_index}")

    # class-aggregated analysis
    au.class_aggregation(features_ranked_all, analysis_run_dir, "all")
    au.class_aggregation(features_ranked_positive, analysis_run_dir, "positive")
    au.class_aggregation(features_ranked_negative, analysis_run_dir, "negative")

    # Visualize the saliency maps and save them as images
    vu.visualize_node_heatmap(node_saliency_map, os.path.join(run_dir, f"node_saliency_map-{i}.png"))
    vu.visualize_edge_heatmap(edge_saliency_map, os.path.join(run_dir, f"edge_saliency_map-{i}.png"))

    au.aa_freq_analysis(zip(os.listdir(pdb_inference_dir), binary_predictions, inference_labels),
                        os.path.join(analysis_run_dir, "amino_acids"))
    au.extract_popular_aas(analysis_run_dir)

    # shap analysis data
    # inference_data_series = pd.Series([inference_tensors])
    # au.perform_shap(model, inference_tensors, os.path.join(shap_run_dir, f"shap.png"))
    #  au.perform_lime(model, inference_graphs, os.path.join(shap_run_dir, f"shap.png"))

    vu.evaluate_model(binary_predictions, inference_labels)
    vu.save_feature_rankings(features_ranked_all, os.path.join(run_dir, "feature_rankings.txt"))
