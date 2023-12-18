# https://iwatobipen.wordpress.com/2022/03/19/build-peptide-from-monomer-library-from-chembl-rdkit-chembl-chemoinformatics/

from rdkit import Chem
from rdkit.Chem import molzip
import copy


def combine_fragments(m1, m2):
    m1 = Chem.Mol(m1)
    m2 = Chem.Mol(m2)
    for atm in m1.GetAtoms():
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
            atm.SetAtomMapNum(1)
    for atm in m2.GetAtoms():
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':  # A što s R3?
            atm.SetAtomMapNum(1)
    return molzip(m1, m2)


def make_peptide(monomerlist):
    monomerlist = copy.deepcopy(monomerlist)
    for idx, monomer in enumerate(monomerlist):
        if Chem.MolToSmiles(monomer).count("*") == 1:
            continue
        if idx == 0:
            res = monomer
        else:
            res = combine_fragments(res, monomer)
    return res


def cap_terminal(m):
    m = Chem.Mol(m)
    n_term = Chem.MolFromSmiles('CC(=O)[*:1]')
    c_term = Chem.MolFromSmiles('CO[*:2]')
    for atm in m.GetAtoms():
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
            atm.SetAtomMapNum(1)
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
            atm.SetAtomMapNum(2)
    res = molzip(m, n_term)
    res = molzip(res, c_term)
    return res