import numpy
import pandas
import torch
from tqdm import tqdm
from mendeleev import get_table
from sklearn import preprocessing
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


elem_feat_names = ['atomic_weight', 'atomic_radius', 'atomic_volume', 'dipole_polarizability',
                   'fusion_heat', 'thermal_conductivity', 'vdw_radius', 'en_pauling']
n_atom_feats = len(elem_feat_names)


def get_elem_feats():
    tb_atom_feats = get_table('elements')
    elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))

    return preprocessing.scale(elem_feats)


def load_dataset(path_user_dataset):
    elem_feats = get_elem_feats()
    list_mols = list()
    id_target = numpy.array(pandas.read_excel(path_user_dataset))

    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(elem_feats, id_target[i, 0], idx=i, target=id_target[i, 1])

        if mol is not None:
            list_mols.append((id_target[i, 0], mol))

    return list_mols

def smiles_to_mol_graph(elem_feats, smiles, idx, target):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        atom_feats = list()
        bonds = list()

        for atom in mol.GetAtoms():
            atom_feats.append(elem_feats[atom.GetAtomicNum() - 1, :])

        for i in range(0, mol.GetNumAtoms()):
            for j in range(0, mol.GetNumAtoms()):
                if adj_mat[i, j] == 1:
                    bonds.append([i, j])

        if len(bonds) == 0:
            return None

        atom_feats = torch.tensor(atom_feats, dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        y = torch.tensor(target, dtype=torch.float).view(1, 1)
        mol_wt = torch.tensor(ExactMolWt(mol), dtype=torch.float).view(1, 1)
        n_rings = torch.tensor(mol.GetRingInfo().NumRings(), dtype=torch.float).view(1, 1)

        return Data(x=atom_feats, y=y, edge_index=bonds, idx=idx, mol_wt=mol_wt, n_rings=n_rings)
    except:
        return None
