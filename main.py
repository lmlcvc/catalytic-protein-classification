from tensorboard.notebook import display

from util import file_utils as fu, graph_utils as gu

import os
import configparser

from stellargraph import datasets
import tensorflow as tf

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

check_setup = config['check_setup']

targets_dir = config['targets_dir']
pdb_catalytic_dir = config['pdb_catalytic_dir']
pdb_non_catalytic_dir = config['pdb_non_catalytic_dir']
graph_catalytic_dir = config['graph_catalytic_dir']
graph_non_catalytic_dir = config['graph_non_catalytic_dir']

pdb_demo_dir = pdb_catalytic_dir.replace("catalytic", "demo")
graph_demo_dir = graph_catalytic_dir.replace("catalytic", "demo")

if __name__ == "__main__":
    # run setup check
    fu.check_setup(check_setup)

    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        fu.generate_targets()

    # demo model generation
    if not os.path.isdir(graph_demo_dir):
        fu.create_folder(graph_demo_dir)
    if not os.listdir(graph_demo_dir):
        for entry in os.listdir(pdb_demo_dir):
            gu.generate_graph(pdb_demo_dir, graph_demo_dir, entry.replace(".pdb", ""))

    # load demo graphs and labels to pass to the model
    graphs = [gu.load_graph(graph_demo_dir, graph) for graph in os.listdir(graph_demo_dir)]
    graph_labels = gu.load_graph_labels()
    gu.graphs_summary(graphs, graph_labels)
