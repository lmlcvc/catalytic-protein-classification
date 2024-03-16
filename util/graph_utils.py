import configparser
import logging
import os.path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges import distance
from graphein.protein.edges.atomic import add_atomic_edges
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from stellargraph import StellarGraph
from functools import partial

import util.file_utils as fu

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_graphein = config['use_graphein']
graph_type = config['graph_type']

targets_dir = config['targets_dir']
graph_dir = config['graph_dir']
categories_dir = config['categories_dir']
file_list_dir = config['file_list_dir']
use_distance_as_weight = config['use_distance_as_weight']

# graphein config
graphein_config = None

if graph_type == "residue":
    edge_construction_funcs = [
        distance.add_aromatic_interactions,
        distance.add_cation_pi_interactions,
        distance.add_aromatic_sulphur_interactions,
        distance.add_disulfide_interactions,
        distance.add_hydrogen_bond_interactions,
        distance.add_hydrophobic_interactions,
        distance.add_ionic_interactions,
        partial(distance.add_distance_threshold, long_interaction_threshold=2, threshold=5.0)
        # intramolecular.pi_stacking
        # intramolecular.salt_bridge,
        # intramolecular.t_stacking,
        # intramolecular.van_der_waals
    ]

    graphein_params_to_change = {"edge_construction_functions": edge_construction_funcs}
    graphein_config = ProteinGraphConfig(**graphein_params_to_change)
elif graph_type == "atom":
    graphein_params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges]}
    graphein_config = ProteinGraphConfig(**graphein_params_to_change)

graphein_config.dict()


def get_distance_matrix(coords):
    diff_tensor = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance_matrix = np.sqrt(np.sum(np.power(diff_tensor, 2), axis=-1))
    return distance_matrix


def replace_categories(df, source_dir, df_type):
    if df_type == "nodes":
        columns = ["residue_name"]
        if graph_type == "atom":
            columns.extend(["atom_type", "element_symbol"])
    elif df_type == "edges":
        columns = ["kind"]
    else:
        raise f"Wrong df type: {df_type}"

    # replace column values with category codes
    for column in columns:
        categories_file = [filename for filename in os.listdir(source_dir) if filename.startswith(column)][0]

        categories_df = pd.read_csv(os.path.join(source_dir, categories_file), header=0)
        categories_dict = categories_df.set_index("value")["category"].to_dict()

        df = df.replace({str(column): categories_dict})

    return df


def prepare_nodes(nodes):
    try:
        # split coords into separate columns
        nodes[['coord_x', 'coord_y', 'coord_z']] = pd.DataFrame(nodes.coords.tolist(), index=nodes.index)

        # remove unnecessary columns
        nodes = nodes.drop(['residue_number', 'chain_id', 'coords', 'meiler'], axis=1)

        residue_names = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
                         "PHE",
                         "PRO", "PYL", "SEC", "SER", "THR", "TRP", "TYR", "VAL"]

        # --- One-hot ---
        # Create new columns for each residue name
        res_names_encoded = pd.DataFrame()
        for residue in residue_names:
            # 1 if that kind was present in edges[kind], otherwise 0
            res_names_encoded[residue] = nodes['residue_name'].apply(
                lambda x: 1 if residue in x.replace("'", "") else 0)
        # ---------------

        # Check for values in 'residue_name' not present in residue_names
        unknown_residues = nodes['residue_name'][~nodes['residue_name'].isin(residue_names)].unique()
        if len(unknown_residues) > 0:
            unknown_residues_str = ', '.join(unknown_residues)
            print(f"Warning: Residue name found in residue_names: {unknown_residues_str}")

        # Apply changes to original df
        nodes = pd.concat([nodes, res_names_encoded], axis=1)

        # remove or transform data depending on graph type
        if graph_type == "atom":
            nodes.atom_type = pd.Categorical(nodes.atom_type)
            nodes['atom_type'] = nodes.atom_type.cat.codes

            nodes.element_symbol = pd.Categorical(nodes.element_symbol)
            nodes['element_symbol'] = nodes.element_symbol.cat.codes
        elif graph_type == "residue":
            nodes = nodes.drop(['atom_type', 'element_symbol', 'residue_name'], axis=1)
        else:
            raise f"Unexpected graph type argument: {graph_type}"

        return nodes

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def prepare_edges(edges):
    try:
        edges['kind'] = edges['kind'].astype(str)
        edges['kind'] = edges['kind'].str.translate({ord('{'): None, ord('}'): None, ord("'"): None})

        edge_kinds = ["aromatic", "aromatic_sulphur", "cation_pi", "disulfide", "hbond", "hydrophobic", "ionic",
                      "protein_bond"]

        # --- Encoding ---
        # Split the values in the 'kind' column and create new columns for each edge kind
        edge_kinds_encoded = pd.DataFrame()
        for kind in edge_kinds:
            # 1 if that kind was present in edges[kind], otherwise 0
            edge_kinds_encoded[kind] = edges['kind'].apply(lambda x: 1 if kind in x.replace("'", "").split(',') else 0)
        # ---------------

        # Warn if there are values in 'kind' that are not present in edge_kinds
        unknown_kinds = set(edges['kind'].str.split(', ').sum()) - set(edge_kinds)
        if unknown_kinds:
            print(f"Warning: Edge kind(s) not in column names: {unknown_kinds}")

        # Apply changes to the original df
        edges = pd.concat([edges, edge_kinds_encoded], axis=1)
        edges = edges.drop(columns=['kind'])

        return edges

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_graph(source_directory, entry, output_directory):
    with open(os.path.join(file_list_dir, "non_duplicate_list.txt")) as f:
        entries = f.read().split('\n')
        if entry not in entries:
            return

    pdb_path = os.path.join(source_directory, f"{entry}.pdb")

    try:
        graph = construct_graph(config=graphein_config,
                                path=pdb_path,
                                pdb_code=entry)
    except:
        logging.error(f"PDB file {entry} failed to transform to graph")
        return

    atom_df = PandasPdb().read_pdb(pdb_path).df['ATOM']
    # adjacency_matrix = nx.to_pandas_adjacency(graph)

    nodes = pd.DataFrame.from_dict(dict(graph.nodes().data()), orient='index')
    edges = nx.to_pandas_edgelist(graph)

    nodes = prepare_nodes(nodes)
    if nodes is None:
        return False

    edges = prepare_edges(edges)
    if edges is None:
        return False

    nodes.to_csv(os.path.join(output_directory, f"{entry}_nodes.csv"))
    edges.to_csv(os.path.join(output_directory, f"{entry}_edges.csv"))
    return True


def standardise_category(category):
    if not isinstance(category, str):
        return category
    elif ',' not in category:
        return category

    return ', '.join(sorted(category.split(', '), key=str.lower))


def store_categories(df, column_list, output_directory, df_type="nodes"):
    fu.create_folder(output_directory)

    for column in column_list:
        categories = df[str(column)].astype('category')
        categories_dict = dict(enumerate(categories.cat.categories))

        categories_df = pd.DataFrame(categories_dict, index=[0]).T
        categories_df.reset_index(inplace=True)
        categories_df.columns = ["category", "value"]
        categories_df = categories_df.reindex(columns=["value", "category"])

        categories_df["category"] = categories_df["category"].apply(standardise_category)

        file_path = os.path.join(output_directory, f"{column}_{df_type}.csv")
        if os.path.exists(file_path):
            categories_df.to_csv(file_path, mode='a', header=False, index=None)
        else:
            categories_df.to_csv(file_path, index=None)


def generate_categories(source_directory, output_directory):
    edges_df = pd.DataFrame()
    nodes_df = pd.DataFrame()

    filenames = sorted(os.listdir(source_directory))
    filename_pairs = [filenames[i:i + 2] for i in range(0, len(filenames), 2)]

    for edges, nodes in filename_pairs:
        if nodes[0:4] != edges[0:4]:
            raise f"PDB names for (nodes, edges) pair do not match: {nodes}, {edges}"

        edges_df = edges_df.append(pd.read_csv(os.path.join(source_directory, edges), index_col=0))
        nodes_df = nodes_df.append(pd.read_csv(os.path.join(source_directory, nodes), index_col=0))

    nodes_categories = ['residue_name']
    edges_categories = ['kind']
    if graph_type == "atom":
        nodes_categories.extend(['atom_type', 'element_symbol'])

    store_categories(nodes_df, nodes_categories, output_directory, df_type="nodes")
    store_categories(edges_df, edges_categories, output_directory, df_type="edges")


def load_graphs(source_directory):
    graphs = []

    filenames = sorted(os.listdir(source_directory))
    filename_pairs = [filenames[i: i + 2] for i in range(0, len(filenames), 2)]

    for edges, nodes in filename_pairs:
        if nodes[0:3] != edges[0:3]:
            raise f"PDB names for (nodes, edges) pair do not match: {nodes}, {edges}"

        edges_df = pd.read_csv(os.path.join(source_directory, edges), index_col=0)
        nodes_df = pd.read_csv(os.path.join(source_directory, nodes), index_col=0)

        if use_distance_as_weight.lower() == 'y':
            graphs.append(StellarGraph(nodes=nodes_df, edges=edges_df, edge_weight_column='distance'))
        else:
            graphs.append(StellarGraph(nodes=nodes_df, edges=edges_df))

    return graphs


def load_graph_labels(filename="targets.txt"):
    with open(os.path.join(targets_dir, filename), "r") as f:
        df = pd.DataFrame([[entry for entry in line.split()] for line in f])
        df.columns = ["index", "label"]

        df = df.set_index(df.columns[0])
        df["label"] = df["label"].astype(float)

        df = df.squeeze()
        return df


def visualise_graph(graph):
    if use_graphein == "Y" or use_graphein == "y":
        visualise_graph_graphein(graph)
    elif use_graphein == "N" or use_graphein == "n":
        visualise_graph_manual(graph)


def visualise_graph_manual(graph):
    plt.figure(figsize=(4, 3), dpi=200)
    nx.draw(graph, pos=nx.kamada_kawai_layout(graph), node_size=50, arrows=False)
    plt.show()


def visualise_graph_graphein(graph):
    p = plotly_protein_structure_graph(
        graph,
        colour_edges_by="kind",
        colour_nodes_by="degree",
        label_node_ids=False,
        plot_title="Peptide backbone graph. Nodes coloured by degree.",
        node_size_multiplier=1
    )
    p.show()


def graphs_summary(graphs, graph_labels):
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )

    print(f"Summary:\n{summary.describe().round(1)}")
    print(f"Graph labels:\n{graph_labels}")
