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
from graphein.protein import add_peptide_bonds, add_hydrogen_bond_interactions
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.molecule.config import MoleculeGraphConfig
from graphein.molecule import atom_type_one_hot, atomic_mass, degree, total_degree, total_valence, explicit_valence, \
    implicit_valence, num_explicit_h, num_implicit_h, total_num_h, num_radical_electrons, formal_charge, hybridization, \
    is_ring, is_isotope, is_aromatic, chiral_tag, add_bond_type, bond_is_aromatic, bond_is_conjugated, bond_stereo

import graphein.protein.graphs as gp
import graphein.molecule.graphs as gm

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']

use_graphein = config['use_graphein']
graph_type = config['graph_type']

targets_dir = config['targets_dir']
graph_dir = config['graph_dir']
cyclic_targets_dir = config['cyclic_targets_dir']
cyclic_inference_dir = config['cyclic_inference_dir']
categories_dir = config['categories_dir']
file_list_dir = config['file_list_dir']
use_distance_as_weight = config['use_distance_as_weight']

# graphein config
graphein_config = None

if graph_type == "residue":
    graphein_params_to_change = {"edge_construction_functions": [add_peptide_bonds, add_hydrogen_bond_interactions]}
    graphein_config = ProteinGraphConfig(**graphein_params_to_change)
elif graph_type == "atom":
    graphein_params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges]}
    graphein_config = ProteinGraphConfig(**graphein_params_to_change)
elif graph_type == "molecule":
    graphein_params_to_change = {
        "add_hs": True,
        "node_metadata_functions": [
            atom_type_one_hot,
            atomic_mass,
            degree,
            total_degree,
            total_valence,
            explicit_valence,
            implicit_valence,
            num_explicit_h,
            num_implicit_h,
            total_num_h,
            num_radical_electrons,
            formal_charge,
            hybridization,
            is_aromatic,
            is_isotope,
            is_ring,
            chiral_tag,
        ],

        "edge_metadata_functions": [
            add_bond_type,
            bond_is_aromatic,
            bond_is_conjugated,
            bond_stereo
        ]
    }
    graphein_config = MoleculeGraphConfig()

graphein_config.dict()


# TODO organise code


def get_distance_matrix(coords):
    diff_tensor = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance_matrix = np.sqrt(np.sum(np.power(diff_tensor, 2), axis=-1))
    return distance_matrix


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


def prepare_nodes_molecular(nodes):
    atom_types = ['C', 'H', 'O', 'N', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B']
    atoms = nodes['rdmol_atom']

    nodes['mass'] = atoms.apply(lambda x: x.GetMass())
    nodes['degree'] = atoms.apply(lambda x: x.GetDegree())
    nodes['total_degree'] = atoms.apply(lambda x: x.GetTotalDegree())
    nodes['total_valence'] = atoms.apply(lambda x: x.GetTotalValence())
    nodes['explicit_valence'] = atoms.apply(lambda x: x.GetExplicitValence())
    nodes['implicit_valence'] = atoms.apply(lambda x: x.GetImplicitValence())
    nodes['num_explicit_h'] = atoms.apply(lambda x: x.GetNumExplicitHs())
    nodes['num_implicit_h'] = atoms.apply(lambda x: x.GetNumImplicitHs())
    nodes['total_num_h'] = atoms.apply(lambda x: x.GetTotalNumHs())
    nodes['num_radical_electrons'] = atoms.apply(lambda x: x.GetNumRadicalElectrons())
    nodes['formal_charge'] = atoms.apply(lambda x: x.GetFormalCharge())
    nodes['hybridization'] = atoms.apply(lambda x: x.GetHybridization())
    nodes['is_aromatic'] = atoms.apply(lambda x: int(x.GetIsAromatic() is True))
    nodes['is_isotope'] = atoms.apply(lambda x: x.GetIsotope())
    nodes['is_ring'] = atoms.apply(lambda x: int(x.IsInRing() is True))
    nodes['chiral_tag'] = atoms.apply(lambda x: x.GetChiralTag())

    for idx, atom_type in enumerate(atom_types):
        nodes[f'is_{atom_type}'] = nodes['atom_type_one_hot'].apply(lambda x: x[idx])

    # remove unnecessary columns
    nodes = nodes.drop(['element', 'coords', 'rdmol_atom', 'atom_type_one_hot'], axis=1)

    return nodes


def prepare_edges_molecular(edges):
    bonds = edges['bond']

    edges['bond_type'] = bonds.apply(lambda x: x.GetBondType())
    edges['bond_is_aromatic'] = bonds.apply(lambda x: int(x.GetIsAromatic() is True))
    edges['bond_is_conjugated'] = bonds.apply(lambda x: int(x.GetIsConjugated() is True))
    edges['bond_stereo'] = bonds.apply(lambda x: x.GetStereo())

    # remove unnecessary columns
    edges = edges.drop(['kind', 'bond'], axis=1)

    return edges


def generate_residue_graph(source_directory, entry, output_directory):
    with open(os.path.join(file_list_dir, "non_duplicate_list.txt")) as f:
        entries = f.read().split('\n')
        if entry not in entries:
            return

    pdb_path = os.path.join(source_directory, f"{entry}.pdb")

    try:
        graph = gp.construct_graph(config=graphein_config, path=pdb_path, pdb_code=entry)
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


def generate_molecule_graph(cycpept, output_directory):
    entry = cycpept["index"]
    smiles = cycpept["SMILES"]

    try:
        graph = gm.construct_graph(config=graphein_config, smiles=smiles)
    except:
        logging.error(f"Entry {entry} with SMILES {smiles} failed to transform to graph")
        return

    nodes = pd.DataFrame.from_dict(dict(graph.nodes().data()), orient='index')
    edges = nx.to_pandas_edgelist(graph)

    nodes = prepare_nodes_molecular(nodes)
    edges = prepare_edges_molecular(edges)

    nodes.to_csv(os.path.join(output_directory, f"CYCPEPT_{entry:04d}_nodes.csv"))
    edges.to_csv(os.path.join(output_directory, f"CYCPEPT_{entry:04d}_edges.csv"))


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

        nodes_df = replace_categories(nodes_df, categories_dir, "nodes")
        edges_df = replace_categories(edges_df, categories_dir, "edges")

        if use_distance_as_weight.lower() == 'y':
            graphs.append(StellarGraph(nodes=nodes_df, edges=edges_df, edge_weight_column='distance'))
        else:
            graphs.append(StellarGraph(nodes=nodes_df, edges=edges_df))

    return graphs


def load_graph_labels(filename="ground_truth.txt"):
    with open(os.path.join(targets_dir, filename), "r") as f:
        df = pd.DataFrame([[entry for entry in line.split()] for line in f])
        df.columns = ["index", "label"]

        df = df.set_index(df.columns[0])
        df["label"] = df["label"].astype(float)  # .astype("category")

        # ?????
        df = df.squeeze()
        return df


def load_cyclic_graphs(source_directory):
    graphs = []

    filenames = sorted(os.listdir(source_directory))
    filename_pairs = [filenames[i: i + 2] for i in range(0, len(filenames), 2)]

    for edges, nodes in filename_pairs:
        edges_id = edges.split('_')[1]
        nodes_id = nodes.split('_')[1]
        if nodes_id != edges_id:
            raise f"IDs for (nodes, edges) pair do not match: {nodes}, {edges}"

        edges_df = pd.read_csv(os.path.join(source_directory, edges), index_col=0)
        nodes_df = pd.read_csv(os.path.join(source_directory, nodes), index_col=0)

        graphs.append(StellarGraph(nodes=nodes_df, edges=edges_df))

    return graphs


def load_cyclic_graph_labels(filename="ground_truth.csv"):
    labels = pd.read_csv(os.path.join(cyclic_targets_dir, filename), index_col=0)
    labels = labels.drop("SMILES", axis=1)
    labels["label"] = labels["label"].astype(float)  # .astype("category")

    # ????? pt. 2
    labels = labels.squeeze()
    return labels


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
