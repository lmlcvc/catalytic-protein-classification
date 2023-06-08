from stellargraph.mapper import PaddedGraphGenerator

from util import file_utils as fu, graph_utils as gu
from model import model as md

import os
import configparser
import pandas as pd
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
pdb_demo_dir = config['pdb_demo_dir']

graph_dir = config['graph_dir']
categories_dir = config['categories_dir']

# suppress "FutureWarning: The default value of regex will change from True to False in a future version." for graph
# generation
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    # run setup check
    fu.check_setup(check_setup)

    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        fu.generate_targets()
        logging.info("Finished target generation")

    # graph generation
    graphs = []
    fu.create_folder(graph_dir)
    if not os.listdir(graph_dir):
        if demo_run == "Y" or demo_run == "y":
            [gu.generate_graph(pdb_demo_dir, entry.replace(".pdb", "")) for entry in os.listdir(pdb_demo_dir)]
            logging.info("Generated demo graphs")

        else:
            [gu.generate_graph(pdb_catalytic_dir, entry.replace(".pdb", "")) for entry in
             os.listdir(pdb_catalytic_dir)]
            logging.info("Generated catalytic graphs")

            [gu.generate_graph(pdb_non_catalytic_dir, entry.replace(".pdb", "")) for entry in
             os.listdir(pdb_non_catalytic_dir)]
            logging.info("Generated non-catalytic graphs")

        gu.generate_categories(graph_dir, categories_dir)

    graphs = gu.load_graphs(graph_dir)

    graph_labels = gu.load_graph_labels()
    gu.graphs_summary(graphs, graph_labels)
    graph_labels.value_counts().to_frame()
    graph_labels = pd.get_dummies(graph_labels, drop_first=True)

    # Adapt graphs to Keras model
    graph_generator = PaddedGraphGenerator(graphs=graphs)
    logging.info("Graphs adapted for model")

    # Create classification models
    model_gcn = md.create_graph_classification_model_gcn(graph_generator)
    logging.info("Created GCN model")
    print(model_gcn.summary())

    model_dcgnn = md.create_graph_classification_model_dcgnn(graph_generator)
    logging.info("Created DCGNN model")
    print(model_dcgnn.summary())
