import numpy as np
import tensorflow as tf
from stellargraph.layer import GraphConvolution
from stellargraph.mapper import PaddedGraphGenerator

from model.model import create_graph_classification_model_gcn
from model.train import train_model
from util import file_utils as fu, graph_utils as gu

import os
import configparser
import logging
import warnings

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

check_setup = config['check_setup']
demo_run = config['demo_run']

targets_dir = config['targets_dir']
pdb_catalytic_dir = config['pdb_catalytic_dir']
pdb_non_catalytic_dir = config['pdb_non_catalytic_dir']
pdb_inference_dir = config['pdb_inference_dir']
pdb_demo_dir = config['pdb_demo_dir']

graph_dir = config['graph_dir']
inference_dir = config['inference_dir']
model_dir = config['model_dir']
categories_dir = config['categories_dir']

# suppress "FutureWarning: The default value of regex will change from True to False in a future version." for graph
# generation
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    # run setup check
    fu.check_setup(check_setup)

    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        fu.generate_targets(pdb_demo_dir)
        logging.info("Finished target generation")

    # graph generation
    graphs = []
    fu.create_folder(graph_dir)
    fu.create_folder(inference_dir)
    if not os.listdir(graph_dir):
        if demo_run == "Y" or demo_run == "y":
            [gu.generate_graph(pdb_demo_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_demo_dir)]
            logging.info("Generated demo graphs")

        else:
            [gu.generate_graph(pdb_catalytic_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_catalytic_dir)]
            logging.info("Generated catalytic graphs")

            [gu.generate_graph(pdb_non_catalytic_dir, entry.replace(".pdb", ""), graph_dir) for entry in
             os.listdir(pdb_non_catalytic_dir)]
            logging.info("Generated non-catalytic graphs")

        [gu.generate_graph(pdb_inference_dir, entry.replace(".pdb", ""), inference_dir) for entry in
         os.listdir(pdb_inference_dir)]
        logging.info("Generated inferenceuation graphs")

        gu.generate_categories(graph_dir, categories_dir)
        logging.info("Generated categories graphs")

    # Adapt graphs to Keras model
    graphs = gu.load_graphs(graph_dir)
    inference_graphs = gu.load_graphs(inference_dir)

    # TODO what connects pdb/graph name to target? (probably order of occurence)
    graph_labels = gu.load_graph_labels()
    gu.graphs_summary(graphs, graph_labels)

    graph_generator = PaddedGraphGenerator(graphs=graphs)

    # TODO modularnije za kad bude više modela (haha)
    fu.create_folder(model_dir)
    if "gcn_model.h5" not in os.listdir(model_dir):
        # Create and train classification models
        model = train_model(graph_generator, graph_labels, epochs=1, folds=2, n_repeats=1)
        print(model.summary())

        # Save the model
        model.save(os.path.join(model_dir, "gcn_model.h5"))
        print("GCN model trained and saved successfully.")
    else:
        with tf.keras.utils.custom_object_scope({'GraphConvolution': GraphConvolution}):
            model = tf.keras.models.load_model(os.path.join(model_dir, "gcn_model.h5"))

    # Prepare your input graph data for inference
    inference_tensors = graph_generator.flow(inference_graphs)

    # Make predictions using the loaded model
    predictions = model.predict(inference_tensors)

    # Convert the predictions to binary class labels (0 or 1)
    binary_predictions = np.round(predictions).astype(int)

    # Print the predictions
    for i, graph in enumerate(inference_graphs):
        prediction = binary_predictions[i][0]
        print(f"Graph {i + 1}: Predicted class - {prediction}")
