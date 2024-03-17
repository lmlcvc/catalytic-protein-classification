# Protein Classification using Graph Convolutional Networks (GCNs)

## Introduction
Catalytic activity is fundamental in bioinformatics, influencing various biological processes. In this study introduces an approach for protein classification based on catalytic activity, using Graph Convolutional Networks (GCNs).

Proteins are represented as graphs using StellarGraph objects derived from the Protein Data Bank (PDB). Each node in the graph represents an amino acid, while edges denote spatial relationships. The adoption of graph-based models is motivated by their ability to capture protein topological patterns. Graphs offer advantages in representing proteins, as they maintain rotation and order invariance. This property allows our model to extract crucial protein features, independent of spatial orientation or sequence.

## Features
### Node
* **Residue Name**: The type of amino acid residue (one-hot encoded).
* **B-Factor**: The B-factor or temperature factor of the residue, indicating the mobility of atoms.
* **X Coordinate**: The X-coordinate of the residue's spatial location.
* **Y Coordinate**: The Y-coordinate of the residue's spatial location.
* **Z Coordinate**: The Z-coordinate of the residue's spatial location.

### Edge
* **Source**: The source node index of the edge.
* **Target**: The target node index of the edge.
* **Distance**: Edge length in Angstroms. Can be derived from edge type or based on threshold (in which case edge has no type).
* **Edge type**: One-hot encoded features representing different types of interactions between nodes, including:
    * Aromatic interactions
    * Aromatic-sulphur interactions
    * Cation-pi interactions
    * Disulfide interactions
    * Hydrogen bond interactions
    * Hydrophobic interactions
    * Ionic interactions
    * Protein bond interactions
