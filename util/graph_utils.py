import configparser
import os.path

import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph
from stellargraph import datasets

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.atomic import add_atomic_edges
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.utils.utils import generate_feature_dataframe

import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

targets_dir = config['targets_dir']
use_graphein = config['use_graphein']

# atom graph
# graphein_params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges]}
# graphein_config = ProteinGraphConfig(**graphein_params_to_change)

# residue graph
graphein_config = ProteinGraphConfig()
graphein_config.dict()
# graphein_params_to_change = {"protein_df_processing_functions": True}
# graphein_config = ProteinGraphConfig(**graphein_params_to_change)

# node_features = pd.DataFrame(
#     {'chain_id', 'residue_name', 'residue_number', 'atom_type', 'element_symbol', 'coords', 'b_factor'}
# )

node_features = pd.DataFrame(
    {"b_factor"}
)


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


def generate_graph(source_directory, destination_directory, entry):
    if use_graphein == "Y" or use_graphein == "y":
        generate_graph_graphein(source_directory, destination_directory, entry)
    elif use_graphein == "N" or use_graphein == "n":
        generate_graph_manual(source_directory, destination_directory, entry)


def generate_graph_manual(source_directory, destination_directory, entry):
    graph = pdb_to_graph(os.path.join(source_directory, f"{entry}.pdb"))
    dgl.save_graphs(os.path.join(destination_directory, f"{entry}.bin"), [graph])


def generate_graph_graphein(source_directory, destination_directory, entry):
    graph = pdb_to_graph_graphein(os.path.join(source_directory, f"{entry}.pdb"), entry)
    # nx.write_graphml(graph, os.path.join(destination_directory, f"{entry}.graphml"))
    nx.write_gexf(graph, os.path.join(destination_directory, f"{entry}.gexf"))


def generate_graph_direct(source_directory, entry):
    pdb_path = os.path.join(source_directory, f"{entry}.pdb")
    graph = construct_graph(config=graphein_config, path=pdb_path, pdb_code=entry)
    print(graph.nodes.data())
    return StellarGraph.from_networkx(graph, node_features=node_features)


def load_graph(source_directory, entry):
    if use_graphein == "Y" or use_graphein == "y":
        load_graph_graphein(source_directory, entry)
    elif use_graphein == "N" or use_graphein == "n":
        load_graph_manual(source_directory, entry)


def load_graph_manual(source_directory, entry):
    graph_list, label_dict = dgl.load_graphs(os.path.join(source_directory, entry))
    return StellarGraph.from_networkx(dgl.to_networkx(graph_list[0]))


def load_graph_graphein(source_directory, entry):
    # graph = nx.read_graphml(os.path.join(source_directory, entry))
    graph = nx.read_gexf(os.path.join(source_directory, entry))
    return StellarGraph.from_networkx(graph)


def load_graph_labels(filename="targets.txt"):
    with open(os.path.join(targets_dir, filename), "r") as f:
        return pd.DataFrame([[entry for entry in line.split()] for line in f])


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
