# https://iwatobipen.wordpress.com/2022/03/19/build-peptide-from-monomer-library-from-chembl-rdkit-chembl-chemoinformatics/
# Modified to close cycle by Alfonso Cabezon Visozo

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
        if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
            atm.SetAtomMapNum(1)
    return molzip(m1, m2)


def make_peptide(monomerlist):
    monomerlist = copy.deepcopy(monomerlist)
    for idx, monomer in enumerate(monomerlist):
        if Chem.MolToSmiles(monomer).count("*") == 1:
            continue
        if idx == 0:
            for atm in monomer.GetAtoms():
                if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R1':
                    atm.SetAtomMapNum(3)  # Where _R2 is found, a 3 is set
            res = monomer
        else:
            if idx == len(monomerlist) - 1:
                for atm in monomer.GetAtoms():
                    if atm.HasProp('atomLabel') and atm.GetProp('atomLabel') == '_R2':
                        atm.SetAtomMapNum(3)  # Where _R2 is found, a 3 is set
            res = combine_fragments(res, monomer)
    return res
