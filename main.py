import networkx as nx
from stellargraph.mapper import PaddedGraphGenerator
from tensorboard.notebook import display

from util import file_utils as fu, graph_utils as gu
from model import model as md

import os
import configparser
import pandas as pd

from stellargraph import datasets
import tensorflow as tf

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

check_setup = config['check_setup']

targets_dir = config['targets_dir']
pdb_catalytic_dir = config['pdb_catalytic_dir']
pdb_non_catalytic_dir = config['pdb_non_catalytic_dir']
graph_dir = config['graph_dir']

if __name__ == "__main__":
    # run setup check
    fu.check_setup(check_setup)

    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        fu.generate_targets()

    # # graph generation
    # if not os.path.isdir(graph_dir):
    #     fu.create_folder(graph_dir)
    # if not os.listdir(graph_dir):
    #     for entry in os.listdir(pdb_catalytic_dir):
    #         print(f"Generating graph for {entry}")
    #         gu.generate_graph(pdb_catalytic_dir, graph_dir, entry.replace(".pdb", ""))
    #     for entry in os.listdir(pdb_non_catalytic_dir):
    #         print(f"Generating graph for {entry}")
    #         gu.generate_graph(pdb_non_catalytic_dir, graph_dir, entry.replace(".pdb", ""))

    # load graphs and labels to pass to the model
    # graphs = [gu.load_graph(graph_dir, graph) for graph in os.listdir(graph_dir)]

    graph = gu.generate_graph_direct(pdb_catalytic_dir, os.listdir(pdb_catalytic_dir)[0].replace(".pdb", ""))
    print(graph.info())
    print(graph.nodes())

    # graphs_catalytic = [gu.generate_graph_direct(pdb_catalytic_dir, entry.replace(".pdb", "")) for entry in
    #                     os.listdir(pdb_catalytic_dir)]
    #
    # graphs_non_catalytic = [gu.generate_graph_direct(pdb_non_catalytic_dir, entry.replace(".pdb", "")) for entry in
    #                         os.listdir(pdb_non_catalytic_dir)]
    #
    # graphs = graphs_catalytic + graphs_non_catalytic

    # graph_labels = gu.load_graph_labels()
    # gu.graphs_summary(graphs, graph_labels)
    # graph_labels.value_counts().to_frame()
    # graph_labels = pd.get_dummies(graph_labels, drop_first=True)
    # # Adapt graphs to Keras model
    # graph_generator = PaddedGraphGenerator(graphs=graphs)

    # Create classification model
    # model = md.create_graph_classification_model_gcn(graph_generator)
    # model = md.create_graph_classification_model_dcgnn(graph_generator)
