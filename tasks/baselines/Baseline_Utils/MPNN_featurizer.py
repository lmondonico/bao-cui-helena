"""
Message Passing Neural Network for featurizing molecular structures in electrolytes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_add_pool
import pandas as pd
from rdkit import Chem
import numpy as np

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "conjugated": [True, False],
}


class ElectrolyteDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.process_data()

    def process_data(self):
        self.processed_data = []
        skipped_rows = 0
        invalid_smiles = 0

        for idx, row in self.df.iterrows():
            sample = {"graphs": [], "concentrations": [], "y": row["LCE"]}

            for i in range(1, 6):
                solvent = row[f"Solvent_{i}"]
                conc = row[f"Conc_Solvent_{i}"]
                if pd.notna(solvent) and pd.notna(conc) and solvent.strip() != "":
                    graph = self.smiles_to_graph(solvent)
                    if graph is not None:
                        sample["graphs"].append(graph)
                        sample["concentrations"].append(float(conc))

            for i in range(1, 4):
                salt = row[f"Salt_{i}"]
                conc = row[f"Conc_Salt_{i}"]
                if pd.notna(salt) and pd.notna(conc) and salt.strip() != "":
                    graph = self.smiles_to_graph(salt)
                    if graph is not None:
                        sample["graphs"].append(graph)
                        sample["concentrations"].append(float(conc))

            if sample["graphs"] and sample["concentrations"]:
                total = sum(sample["concentrations"])
                sample["concentrations"] = [c / total for c in sample["concentrations"]]
                self.processed_data.append(sample)
            else:
                skipped_rows += 1

        print(f"Processed {len(self.processed_data)} samples")
        print(f"Skipped {skipped_rows} rows due to missing or invalid data")
        if invalid_smiles > 0:
            print(f"Failed to parse {invalid_smiles} SMILES strings")

    def smiles_to_graph(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.invalid_smiles += 1
                print(f"Failed to parse SMILES: {smiles}")
                return None
        except Exception as e:
            self.invalid_smiles += 1
            print(f"Error processing SMILES {smiles}: {str(e)}")
            return None

        x = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetChiralTag()),
                atom.GetTotalNumHs(),
                int(atom.GetHybridization()),
            ]
            x.append(features)
        x = torch.tensor(x, dtype=torch.float)

        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices.extend([[i, j], [j, i]])

            features = [int(bond.GetBondType()), bond.GetIsConjugated()]
            edge_features.extend([features, features])

        if not edge_indices:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        sample = self.processed_data[idx]
        return (
            sample["graphs"],
            torch.tensor(sample["concentrations"], dtype=torch.float32),
            torch.tensor(sample["y"], dtype=torch.float32),
        )


class MPNNConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr="add")

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_gru = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = self.update_gru(aggr_out, x)

        return out

    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)

        return self.message_mlp(msg_input)


class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_layers):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList(
            [MPNNConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.mol_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        x = global_add_pool(x, batch)

        x = self.mol_embed(x)

        return x


class ElectrolyteModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_layers):
        super().__init__()
        self.mpnn = MPNN(node_dim, edge_dim, hidden_dim, n_layers)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, graphs, concentrations):
        batch = Batch.from_data_list(graphs)

        mol_embeddings = self.mpnn(batch)

        batch_size = concentrations.shape[0]
        max_mols = concentrations.shape[1]
        mol_embeddings = mol_embeddings.view(batch_size, max_mols, -1)

        concentrations = concentrations.unsqueeze(-1)
        weighted_sum = torch.sum(mol_embeddings * concentrations, dim=1)

        return self.predictor(weighted_sum).squeeze(-1)


def collate_fn(samples):
    max_mols = max(len(sample[0]) for sample in samples)
    graphs_list = []
    labels = []
    batched_concentrations = []

    for sample_graphs, sample_conc, label in samples:
        graphs_list.extend(sample_graphs)

        if len(sample_conc) < max_mols:
            padded_conc = torch.zeros(max_mols, dtype=torch.float32)
            padded_conc[: len(sample_conc)] = sample_conc
            batched_concentrations.append(padded_conc)
        else:
            batched_concentrations.append(sample_conc)

        labels.append(label)

    return graphs_list, torch.stack(batched_concentrations), torch.stack(labels)
