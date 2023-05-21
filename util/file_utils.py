import os
import configparser
import pandas as pd
import warnings

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

tables_dir = config['tables_dir']
targets_dir = config['targets_dir']

def generate_targets():
    entries_list = []

    for table in os.listdir(tables_dir):
        df = pd.read_excel(os.path.join(tables_dir, table))

        for entry in df['Entry'].tolist():
            if 'non-catalytic' in table:
                entries_list.append(entry + " 0")
            elif 'catalytic' in table:
                entries_list.append(entry + " 1")
            else:
                warnings.warn(f"Unexpected table name: {table}")

    targets_file = open(os.path.join(targets_dir, "targets.txt"), "w")
    targets_file.write("\n".join(entry for entry in entries_list))
