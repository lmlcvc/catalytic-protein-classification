import util.file_utils as fu

import configparser
import os.path

import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
from stellargraph import StellarGraph
import logging

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.atomic import add_atomic_edges
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_graphein = config['use_graphein']
graph_type = config['graph_type']

targets_dir = config['targets_dir']
graph_dir = config['graph_dir']
categories_dir = config['categories_dir']
file_list_dir = config['file_list_dir']

# graphein config
graphein_config = None
graphein_params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges]}
# graphein_params_to_change = {"protein_df_processing_functions": True}

if graph_type == "residue":
    graphein_config = ProteinGraphConfig()
elif graph_type == "atom":
    graphein_config = ProteinGraphConfig(**graphein_params_to_change)

graphein_config.dict()

# TODO organise code


def get_distance_matrix(coords):
    diff_tensor = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance_matrix = np.sqrt(np.sum(np.power(diff_tensor, 2), axis=-1))
    return distance_matrix


def pdb_to_graph(pdb_path, distance_threshold=6.0, contain_b_factor=True):
    atom_df = PandasPdb().read_pdb(pdb_path)
    atom_df = atom_df.df['ATOM']

    residue_df = atom_df.groupby('residue_number', as_index=False)[
        ['x_coord', 'y_coord', 'z_coord', 'b_factor']].mean().sort_values('residue_number')

    coords = residue_df[['x_coord', 'y_coord', 'z_coord']].values
    distance_matrix = get_distance_matrix(coords)
    adj = distance_matrix < distance_threshold

    u, v = np.nonzero(adj)
    u, v = torch.from_numpy(u), torch.from_numpy(v)
    graph = dgl.graph((u, v), num_nodes=len(coords))

    if contain_b_factor:
        b_factor = torch.from_numpy(residue_df['b_factor'].values)
        graph.ndata['b_factor'] = b_factor

    return graph


def pdb_to_graph_graphein(pdb_path, entry):
    graph = construct_graph(config=graphein_config, path=pdb_path, pdb_code=entry)
    return graph


def replace_categories(df, source_dir, df_type):
    if df_type == "nodes":
        columns = ["residue_name"]
        if graph_type == "atom":
            columns.extend(["atom_type", "element_symbol"])
    elif df_type == "edges":  # TODO maybe encode source and target as well (preferably same as nodes encoding)
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
    # split coords into separate columns
    nodes[['coord_x', 'coord_y', 'coord_z']] = pd.DataFrame(nodes.coords.tolist(), index=nodes.index)

    # remove unnecessary columns
    nodes = nodes.drop(['chain_id', 'coords', 'meiler'], axis=1)

    # remove or transform data depending on graph type
    if graph_type == "atom":
        nodes.atom_type = pd.Categorical(nodes.atom_type)
        nodes['atom_type'] = nodes.atom_type.cat.codes

        nodes.element_symbol = pd.Categorical(nodes.element_symbol)
        nodes['element_symbol'] = nodes.element_symbol.cat.codes
    elif graph_type == "residue":
        nodes = nodes.drop(['atom_type', 'element_symbol'], axis=1)
    else:
        raise f"Unexpected graph type argument: {graph_type}"

    return nodes


def prepare_edges(edges):
    edges['kind'] = edges['kind'].astype(str)
    edges['kind'] = edges['kind'].str.translate({ord('{'): None, ord('}'): None, ord("'"): None})

    return edges


def generate_graph(source_directory, entry, output_directory):
    with open(os.path.join(file_list_dir, "non_duplicate_list.txt")) as f:
        entries = f.read().split('\n')
        if entry not in entries:
            return

    pdb_path = os.path.join(source_directory, f"{entry}.pdb")

    try:
        graph = construct_graph(config=graphein_config, path=pdb_path, pdb_code=entry)
    except:
        logging.error(f"PDB file {entry} failed to transform to graph")
        return

    atom_df = PandasPdb().read_pdb(pdb_path).df['ATOM']
    # adjacency_matrix = nx.to_pandas_adjacency(graph)

    nodes = pd.DataFrame.from_dict(dict(graph.nodes().data()), orient='index')
    edges = nx.to_pandas_edgelist(graph)

    nodes = prepare_nodes(nodes)
    edges = prepare_edges(edges)

    nodes.to_csv(os.path.join(output_directory, f"{entry}_nodes.csv"))
    edges.to_csv(os.path.join(output_directory, f"{entry}_edges.csv"))


def store_categories(df, column_list, output_directory, df_type="nodes"):
    fu.create_folder(output_directory)

    for column in column_list:
        categories = df[str(column)].astype('category')
        categories_dict = dict(enumerate(categories.cat.categories))

        categories_df = pd.DataFrame(categories_dict, index=[0]).T
        categories_df.reset_index(inplace=True)
        categories_df.columns = ["category", "value"]
        categories_df = categories_df.reindex(columns=["value", "category"])
        categories_df.to_csv(os.path.join(output_directory, f"{column}_{df_type}.csv"), index=None)


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

        nodes_df = replace_categories(nodes_df, categories_dir, "nodes")
        edges_df = replace_categories(edges_df, categories_dir, "edges")

        graphs.append(StellarGraph(nodes=nodes_df, edges=edges_df))

    return graphs


def load_graph_labels(filename="targets.txt"):
    with open(os.path.join(targets_dir, filename), "r") as f:
        df = pd.DataFrame([[entry for entry in line.split()] for line in f])
        df.columns = ["index", "label"]

        df = df.set_index(df.columns[0])
        df["label"] = df["label"].astype(float)  # .astype("category")

        # ?????
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
