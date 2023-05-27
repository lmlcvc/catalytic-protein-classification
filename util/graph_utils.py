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

import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']


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


def generate_graph(source_directory, destination_directory, entry):
    graph = pdb_to_graph(os.path.join(source_directory, f"{entry}.pdb"))
    dgl.save_graphs(os.path.join(destination_directory, f"{entry}.bin"), [graph])


def load_graph(source_directory, entry):
    graph_list, label_dict = dgl.load_graphs(os.path.join(source_directory, f"{entry}.bin"))
    return graph_list[0]


def visualise_graph(graph):
    nx_graph = dgl.to_networkx(graph)
    plt.figure(figsize=(4, 3), dpi=200)
    nx.draw(nx_graph, pos=nx.kamada_kawai_layout(nx_graph), node_size=50, arrows=False)
    plt.show()


def graphs_summary(graphs, graph_labels):
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )

    print(f"Summary:\n{summary.describe().round(1)}")
    print(f"Graph labels:\n{graph_labels.value_counts().to_frame()}")
