import numpy as np
import tensorflow as tf
from stellargraph.layer import GraphConvolution
from stellargraph.mapper import PaddedGraphGenerator

from model.train import train_model
from model.model import get_gradients, visualize_grad_cam
from util import file_utils as fu, graph_utils as gu
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

# suppress "FutureWarning: The default value of regex will change from True to False in a future version." for graph
# generation
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    # run setup check
    fu.check_setup(check_setup)

    if one_per_entry.lower() == "y":
        fu.pick_best()

    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        if demo_run == "Y" or demo_run == "y":
            fu.generate_targets([pdb_demo_dir])
        else:
            fu.generate_targets([pdb_catalytic_dir, pdb_non_catalytic_dir])
        logging.info("Finished target generation")

    # Human-readable ground truth files
    fu.generate_ground_truth(pdb_inference_dir)

    # graph generation
    graphs = []
    fu.create_folder(graph_dir)
    fu.create_folder(inference_dir)

    if demo_run == "Y" or demo_run == "y":
        if not os.listdir(demo_graph_dir):
            [gu.generate_graph(pdb_demo_dir, entry.replace(".pdb", ""), demo_graph_dir) for entry in
             os.listdir(pdb_demo_dir)]
            logging.info("Generated demo graphs")

        gu.generate_categories(demo_graph_dir, categories_dir)
        logging.info("Generated demo categories graphs")
    else:
        if not os.listdir(graph_dir):
            [gu.generate_graph(pdb_catalytic_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_catalytic_dir)]
            logging.info("Generated catalytic graphs")

            [gu.generate_graph(pdb_non_catalytic_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_non_catalytic_dir)]
            logging.info("Generated non-catalytic graphs")

        gu.generate_categories(graph_dir, categories_dir)
        logging.info("Generated categories graphs")

    if not os.listdir(inference_dir):
        [gu.generate_graph(pdb_inference_dir, entry.replace(".pdb", ""), inference_dir) for entry in
         os.listdir(pdb_inference_dir)]
        logging.info("Generated inference graphs")

    # Adapt graphs to Keras model
    if demo_run == "Y" or demo_run == "y":
        graphs = gu.load_graphs(demo_graph_dir)
    else:
        graphs = gu.load_graphs(graph_dir)
    inference_graphs = gu.load_graphs(inference_dir)

    if not os.path.isdir(visualization_dir):
        os.mkdir(visualization_dir)

    # TODO what connects pdb/graph name to target? (probably order of occurence)
    graph_labels = gu.load_graph_labels()
    gu.graphs_summary(graphs, graph_labels)

    graph_generator = PaddedGraphGenerator(graphs=graphs)

    # TODO modularnije za kad bude više modela (haha)
    fu.create_folder(model_dir)
    if "gcn_model.h5" not in os.listdir(model_dir):
        # Create and train classification models
        model = train_model(graph_generator, graph_labels, epochs=200, folds=10, n_repeats=5)
        print(model.summary())

        # Save the model
        model.save(os.path.join(model_dir, "gcn_model.h5"))
        print("GCN model trained and saved successfully.")
    else:
        with tf.keras.utils.custom_object_scope({'GraphConvolution': GraphConvolution}):
            model = tf.keras.models.load_model(os.path.join(model_dir, "gcn_model.h5"))
            print(model.summary())

    inference_labels = gu.load_graph_labels("inference_truth.txt")
    # Prepare input graph data for inference
    # inference_tensors = graph_generator.flow(inference_graphs, weighted=True, targets=inference_labels)
    inference_generator = PaddedGraphGenerator(graphs=inference_graphs)
    inference_tensors = inference_generator.flow(inference_graphs, weighted=True, targets=inference_labels)

    # Make predictions using the loaded model
    predictions = model.predict(inference_tensors)

    # Convert the predictions to binary class labels (0 or 1)
    binary_predictions = np.round(predictions).astype(int)

    # Compute and visualize Grad-CAM heatmaps for each sample in the inference dataset
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.mkdir(os.path.join(visualization_dir, f"{run_timestamp}"))
    run_dir = os.path.join(os.path.join(visualization_dir, f"{run_timestamp}"))

    for i, graph in enumerate(inference_graphs):
        prediction = binary_predictions[i][0]
        print(f"Graph {i + 1}: Predicted class - {prediction}")

        # Get the input features for the sample
        inputs = inference_tensors[i][0]
        if inputs is None:
            print(f"Skipping graph {i + 1} due to None input.")
            continue

        # Compute the gradients for the selected sample
        gradients = get_gradients(model, inputs)
        print(gradients)

        # Calculate importance weights by averaging gradients across channels
        importance_weights = np.mean(gradients, axis=-1)  # Assuming the last axis represents the channels

        # Normalize importance weights to [0, 1] range
        importance_weights = (importance_weights - np.min(importance_weights)) / (
                    np.max(importance_weights) - np.min(importance_weights))

        # Visualize the Grad-CAM heatmap
        visualize_grad_cam(importance_weights, os.path.join(run_dir, f"gradcam_{i + 1}.png"))
