from util import file_utils as fu, graph_utils as gu

import os
import configparser

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

    # demo graph generation
    fu.create_folder(graph_demo_dir)
    for entry in os.listdir(pdb_demo_dir):
        gu.generate_graph(pdb_demo_dir, graph_demo_dir, entry.replace(".pdb", ""))
