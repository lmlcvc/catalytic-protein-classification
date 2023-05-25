from util import file_utils as fu

import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

targets_dir = config['targets_dir']

if __name__ == "__main__":
    fu.check_filecount()
    # check if files have been transformed
    if not os.path.isdir(targets_dir) or not os.listdir(targets_dir):
        fu.generate_targets()
