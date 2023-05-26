import os
import configparser
import pandas as pd
import warnings

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

tables_dir = config['tables_dir']
targets_dir = config['targets_dir']
file_list_dir = config['file_list_dir']

pdb_catalytic_dir = config['pdb_catalytic_dir']
pdb_non_catalytic_dir = config['pdb_non_catalytic_dir']


def create_folder(output_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    return


def check_setup(check_setup):
    if check_setup == "Y" or check_setup == "y":
        run_full_setup_check()
    elif check_setup == "N" or check_setup == "n":
        pass
    else:
        check_setup = input("Execute setup check? [Y/n] ")
        if check_setup != "Y" and check_setup != "y":
            pass
        else:
            run_full_setup_check()


def run_full_setup_check():
    check_filecount()
    # will add function calls when implemented


def check_filecount():
    """ Check which intended files from initial dataset are not present or failed to download """

    catalytic_files = os.listdir(pdb_catalytic_dir)
    non_catalytic_files = os.listdir(pdb_non_catalytic_dir)
    missing_catalytic = 0
    missing_non_catalytic = 0

    for list_file in os.listdir(file_list_dir):
        with open(os.path.join(file_list_dir, list_file)) as f:
            entries = f.read().split(',')

            for entry in entries:
                if 'non-catalytic' in list_file:
                    if f"{entry}.pdb" not in non_catalytic_files:
                        warnings.warn(f"Missing: non-catalytic {entry}")
                        missing_non_catalytic += 1
                elif 'catalytic' in list_file:
                    if f"{entry}.pdb" not in catalytic_files:
                        warnings.warn(f"Missing: catalytic {entry}")
                        missing_catalytic += 1
                else:
                    warnings.warn(f"Unexpected file name: {list_file}")

        print(f"Missing:\ncatalytic = {missing_catalytic}\n"
              f"non-catalytic = {missing_non_catalytic}\n"
              f"total = {missing_catalytic + missing_non_catalytic}")


def generate_targets():
    """
    Generate files with binary target values for each protein entry name
        0 - non-catalytic
        1 - catalytic
    """
    entries_list = []

    for table in os.listdir(tables_dir):
        df = pd.read_excel(os.path.join(tables_dir, table))

        for entry in df['Entry'].tolist():
            if 'non-catalytic' in table:
                entries_list.append(f"{entry} 0")
            elif 'catalytic' in table:
                entries_list.append(f"{entry} 1")
            else:
                warnings.warn(f"Unexpected table name: {table}")

    targets_file = open(os.path.join(targets_dir, "targets.txt"), "w")
    targets_file.write("\n".join(entry for entry in entries_list))
