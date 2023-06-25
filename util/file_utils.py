import os
import configparser
import pandas as pd
import warnings

from biopandas.pdb import PandasPdb

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
    list_non_duplicates()
    pick_best()
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


def list_non_duplicates():
    print("Dropping duplicate pdbs")

    pdb_list = []
    duplicate_list = []

    for table in os.listdir(tables_dir):
        df = pd.read_excel(os.path.join(tables_dir, table))

        for pdbs in df['PDB'].tolist():
            pdb_field = [pdb for pdb in pdbs.split(";") if pdb != ""]
            for pdb in pdb_field:
                if pdb not in pdb_list and pdb not in duplicate_list:
                    pdb_list.append(pdb)
                elif pdb in pdb_list:
                    pdb_list.remove(pdb)
                    duplicate_list.append(pdb)

    pdbs_file = open(os.path.join(file_list_dir, "non_duplicate_list.txt"), "w")
    pdbs_file.write("\n".join(pdb for pdb in pdb_list))


def pick_best():
    print("Selecting highest quality pdb for each entry")
    pdb_list = []

    for table in os.listdir(tables_dir):
        df = pd.read_excel(os.path.join(tables_dir, table))

        for index, row in df.iterrows():
            min_rfree = 1
            best_pdb = None
            for pdb in row['PDB'].split(";"):
                if pdb == "":
                    continue

                try:
                    if 'non-catalytic' in table:
                        pdb_df = PandasPdb().read_pdb(
                            os.path.join(pdb_non_catalytic_dir, f"{pdb}.pdb")
                        ).df['OTHERS']
                    elif 'catalytic' in table:
                        pdb_df = PandasPdb().read_pdb(
                            os.path.join(pdb_catalytic_dir, f"{pdb}.pdb")
                        ).df['OTHERS']
                    else:
                        warnings.warn(f"Unexpected table name: {table}")

                except FileNotFoundError:
                    continue

                for line in pdb_df['entry'].tolist():
                    if "R VALUE" in line and "(WORKING SET)" in line:
                        if "null" not in line.lower():
                            r_value = eval(line.split(":")[1])
                            if r_value < min_rfree:
                                min_rfree = r_value
                                best_pdb = pdb
                                break

            if best_pdb is not None:
                pdb_list.append(best_pdb)

    pdbs_file = open(os.path.join(file_list_dir, "best_pdb_list.txt"), "w")
    pdbs_file.write("\n".join(pdb for pdb in pdb_list))


def generate_targets(pdb_source_directories):
    """
    Generate files with binary target values for each protein entry name
        0 - non-catalytic
        1 - catalytic
    """
    lines = []

    for table in os.listdir(tables_dir):
        df = pd.read_excel(os.path.join(tables_dir, table))

        for pdbs in df['PDB'].tolist():
            pdb_list = [pdb for pdb in pdbs.split(";") if pdb != ""]
            for pdb in pdb_list:
                if 'non-catalytic' in table:
                    lines.append(f"{pdb} 0")
                elif 'catalytic' in table:
                    lines.append(f"{pdb} 1")
                else:
                    warnings.warn(f"Unexpected table name: {table}")

    pdb_target_list = []
    for pdb_source_directory in pdb_source_directories:
        for pdb in os.listdir(pdb_source_directory):
            pdb_target_list.append(pdb.replace(".pdb", ""))
    pdb_target_list.sort(key=str.lower)

    lines_filtered = []
    for line in lines:
        if line[0:4] in pdb_target_list and line not in lines_filtered:
            lines_filtered.append(line)

    create_folder(targets_dir)
    targets_file = open(os.path.join(targets_dir, "targets.txt"), "w")
    targets_file.write("\n".join(line for line in lines_filtered))


def generate_ground_truth(pdb_source_directory):
    """
    Generate files with binary target values for each protein entry name
        0 - non-catalytic
        1 - catalytic
    """
    lines = []

    for table in os.listdir(tables_dir):
        df = pd.read_excel(os.path.join(tables_dir, table))

        for pdbs in df['PDB'].tolist():
            pdb_list = [pdb for pdb in pdbs.split(";") if pdb != ""]
            for pdb in pdb_list:
                if 'non-catalytic' in table:
                    lines.append(f"{pdb} 0")
                elif 'catalytic' in table:
                    lines.append(f"{pdb} 1")
                else:
                    warnings.warn(f"Unexpected table name: {table}")

    pdb_target_list = [pdb.replace(".pdb", "") for pdb in os.listdir(pdb_source_directory)]
    pdb_target_list.sort(key=str.lower)

    lines_filtered = []
    for line in lines:
        if line[0:4] in pdb_target_list and line not in lines_filtered:
            lines_filtered.append(line)

    create_folder(targets_dir)
    targets_file = open(os.path.join(targets_dir, "inference_truth.txt"), "w")
    targets_file.write("\n".join(line for line in lines_filtered))
