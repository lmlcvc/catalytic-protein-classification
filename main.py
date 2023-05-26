from util import file_utils as fu
from graph import generate_graph as gg

import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

check_setup = config['check_setup']

targets_dir = config['targets_dir']

if __name__ == "__main__":
    # run setup check
    fu.check_setup(check_setup)

    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        fu.generate_targets()

    gg.generate_graph()
