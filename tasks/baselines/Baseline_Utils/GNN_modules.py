"""
Baseline GNN architectures for comparison with HELENA.
Implements simplified versions of GCN, GAT, and MPNN for molecular formulation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    NNConv,
    global_mean_pool,
)
from torch_geometric.nn.models import AttentiveFP
from rdkit import Chem
import regex as re
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")


class MoleculeEncoder:
    """Utility class to convert SMILES to PyTorch Geometric graph data using only 2D information"""

    def __init__(self):
        pass

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            node_features.append(features)

        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices.append([i, j])
            edge_indices.append([j, i])

            bond_type = bond.GetBondTypeAsDouble()
            is_conj = int(bond.GetIsConjugated())
            is_in_ring = int(bond.IsInRing())

            edge_feature = [bond_type, is_conj, is_in_ring]

            edge_features.append(edge_feature)
            edge_features.append(edge_feature)

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


class MolformerMoleculeEncoder:
    """
    Molecule encoder for Molformer that converts SMILES to tokenized sequences.

    This implementation faithfully reproduces the tokenization and vocabulary
    handling from the original Molformer paper, using the same regex pattern
    and vocabulary file.
    """

    def __init__(self, max_length=202):
        self.max_length = max_length

        vocab_path = os.path.join(
            os.path.dirname(__file__),
            "Molformer",
            "training",
            "pubchem_canon_zinc_final_vocab_sorted.pth",
        )
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Molformer vocabulary file not found at {vocab_path}"
            )

        self.vocab_encoder = torch.load(
            vocab_path, map_location="cpu", weights_only=False
        )

        self.char2id = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<mask>": 3}
        self.id2char = {0: "<bos>", 1: "<eos>", 2: "<pad>", 3: "<mask>"}

        pos = 0
        for key, value in self.vocab_encoder.items():
            self.char2id[key] = pos + 4
            self.id2char[pos + 4] = key
            pos += 1
        self.char2id["<unk>"] = pos + 4
        self.id2char[pos + 4] = "<unk>"

        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

        self.pad = self.char2id["<pad>"]
        self.bos = self.char2id["<bos>"]
        self.eos = self.char2id["<eos>"]
        self.unk = self.char2id["<unk>"]

    def tokenize_smiles(self, smiles):
        tokens = self.regex.findall(smiles)
        if "".join(tokens) != smiles:
            tokens = [token if token in self.char2id else "<unk>" for token in tokens]
        return tokens

    def encode_smiles(self, smiles):
        tokens = self.tokenize_smiles(smiles)

        tokens = ["<bos>"] + tokens + ["<eos>"]

        token_ids = []
        for token in tokens:
            if token in self.char2id:
                token_ids.append(self.char2id[token])
            else:
                token_ids.append(self.unk)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length - 1] + [self.eos]

        return torch.tensor(token_ids, dtype=torch.long)

    def encode_molecule(self, smiles):
        return self.encode_smiles(smiles)

    def get_vocab_size(self):
        return len(self.char2id)


class MolformerModel(nn.Module):
    """
    Molformer-inspired model for molecular property prediction.

    LIMITATIONS: This implementation uses PyTorch's standard transformer
    instead of the original's linear attention + rotary embeddings due to
    dependency constraints. Key differences from original:
    - Uses standard multi-head attention instead of linear attention
    - Uses learned positional embeddings instead of rotary embeddings
    - Missing fast_transformers optimizations
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim=768,
        num_layers=6,
        num_heads=12,
        max_length=202,
        dropout=0.1,
    ):
        super(MolformerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)

        self.pos_emb = None

        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.lang_model = self.lm_layer(hidden_dim, vocab_size)

    class lm_layer(nn.Module):
        """Language model head - exactly as in original"""

        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)

        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def create_padding_mask(self, x, pad_token_id=2):
        return x == pad_token_id

    def forward(self, x):
        batch_size, seq_len = x.size()

        padding_mask = self.create_padding_mask(x)

        token_embeddings = self.tok_emb(x)

        embeddings = self.drop(token_embeddings)

        transformer_output = self.transformer(
            embeddings, src_key_padding_mask=padding_mask
        )

        mask = (~padding_mask).float().unsqueeze(-1)
        masked_output = transformer_output * mask
        pooled_output = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled_output


class GCNModel(nn.Module):
    """Simple Graph Convolutional Network (GCN) for molecular property prediction"""

    def __init__(self, node_features_dim=6, hidden_dim=64, out_dim=1, num_layers=3):
        super(GCNModel, self).__init__()

        self.node_embedding = nn.Linear(node_features_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data["x"], data["edge_index"]

        x = self.node_embedding(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(
            x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = self.readout_mlp(x)

        return x


class GATModel(nn.Module):
    """Graph Attention Network (GAT) for molecular property prediction"""

    def __init__(
        self, node_features_dim=6, hidden_dim=64, out_dim=1, num_layers=3, heads=4
    ):
        super(GATModel, self).__init__()

        self.node_embedding = nn.Linear(node_features_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data["x"], data["edge_index"]

        x = self.node_embedding(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(
            x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = self.readout_mlp(x)

        return x


class MPNNModel(nn.Module):
    """Message Passing Neural Network (MPNN) for molecular property prediction"""

    def __init__(
        self,
        node_features_dim=6,
        edge_features_dim=3,
        hidden_dim=32,
        out_dim=1,
        num_layers=3,
    ):
        super(MPNNModel, self).__init__()

        self.node_embedding = nn.Linear(node_features_dim, hidden_dim)

        self.edge_network = nn.Sequential(
            nn.Linear(edge_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(NNConv(hidden_dim, hidden_dim, self.edge_network))

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]

        x = self.node_embedding(x)
        x = F.relu(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(
            x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = self.readout_mlp(x)

        return x


class AttentiveFPModel(nn.Module):
    """AttentiveFP model for molecular property prediction"""

    def __init__(
        self,
        node_features_dim=6,
        edge_features_dim=3,
        hidden_dim=64,
        out_dim=1,
        num_layers=3,
        num_timesteps=2,
        dropout=0.0,
    ):
        super(AttentiveFPModel, self).__init__()

        self.attentive_fp = AttentiveFP(
            in_channels=node_features_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            edge_dim=edge_features_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, data):
        x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]

        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.attentive_fp(x, edge_index, edge_attr, batch)

        return x


class BaselineFormulationModel(nn.Module):
    """
    Baseline model for formulation property prediction using multiple molecular graphs
    and concentration weighting for the final embedding.
    """

    def __init__(
        self,
        gnn_type="GCN",
        node_features_dim=6,
        edge_features_dim=3,
        hidden_dim=128,
        out_dim=1,
        num_layers=3,
        heads=4,
        max_molecules=8,
        molformer_encoder=None,
    ):
        super(BaselineFormulationModel, self).__init__()

        self.gnn_type = gnn_type
        self.max_molecules = max_molecules
        self.hidden_dim = hidden_dim

        if gnn_type == "GCN":
            self.gnn = GCNModel(node_features_dim, hidden_dim, hidden_dim, num_layers)
        elif gnn_type == "GAT":
            self.gnn = GATModel(
                node_features_dim, hidden_dim, hidden_dim, num_layers, heads
            )
        elif gnn_type == "MPNN":
            self.gnn = MPNNModel(
                node_features_dim, edge_features_dim, hidden_dim, hidden_dim, num_layers
            )
        elif gnn_type == "AttentiveFP":
            self.gnn = AttentiveFPModel(
                node_features_dim, edge_features_dim, hidden_dim, hidden_dim, num_layers
            )
        elif gnn_type == "Molformer":
            if molformer_encoder is None:
                raise ValueError("molformer_encoder is required for Molformer model")
            self.molformer_encoder = molformer_encoder
            vocab_size = molformer_encoder.get_vocab_size()
            self.gnn = MolformerModel(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=heads,
                max_length=molformer_encoder.max_length,
            )
        else:
            raise ValueError(
                f"GNN type {gnn_type} not supported. Use 'GCN', 'GAT', 'MPNN', 'AttentiveFP', 'Molformer'"
            )

        # Encoders for current and method
        self.current_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        )
        self.method_encoder = nn.Embedding(2, hidden_dim)

        # Final MLP: molecule embedding + current + method
        self.formulation_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, mol_graphs, concentrations, currents, methods):
        batch_size = len(mol_graphs)
        device = next(self.parameters()).device

        if self.gnn_type == "AttentiveFP":
            embedding_dim = self.gnn.attentive_fp.out_channels
        elif self.gnn_type == "Molformer":
            embedding_dim = self.gnn.hidden_dim
        else:
            embedding_dim = self.gnn.readout_mlp[0].in_features

        formulation_embeddings = torch.zeros(batch_size, embedding_dim, device=device)

        for b in range(batch_size):
            valid_mols = len(mol_graphs[b])
            mol_embeddings = []

            for mol_idx in range(valid_mols):
                if self.gnn_type == "Molformer":
                    smiles = mol_graphs[b][mol_idx]
                    tokens = (
                        self.molformer_encoder.encode_smiles(smiles)
                        .unsqueeze(0)
                        .to(device)
                    )
                    mol_emb = self.gnn(tokens).squeeze(0)
                else:
                    mol_data = mol_graphs[b][mol_idx]
                    mol_data = {k: v.to(device) for k, v in mol_data.items()}
                    mol_emb = self.gnn(mol_data)

                mol_embeddings.append(mol_emb)

            if mol_embeddings:
                mol_embeddings = torch.stack(mol_embeddings, dim=0).squeeze(1)

                concs = concentrations[b, :valid_mols].to(device)

                weighted_embeddings = mol_embeddings * concs.unsqueeze(-1)

                formulation_embeddings[b] = weighted_embeddings.sum(dim=0)

        # Encode current (log-transformed) and method
        current_emb = self.current_encoder(currents.unsqueeze(-1))
        method_emb = self.method_encoder(methods.long())

        # Concatenate all embeddings
        combined = torch.cat([formulation_embeddings, current_emb, method_emb], dim=-1)

        predictions = self.formulation_mlp(combined)

        return predictions


def preprocess_formulation_batch(batch, encoder):
    batch_size = len(batch["smiles"])
    max_molecules = len(batch["smiles"][0])

    mol_graphs = []
    all_concentrations = []

    for b in range(batch_size):
        formulation_graphs = []
        formulation_concs = []

        for i in range(max_molecules):
            if batch["smiles"][b][i] is not None:
                graph = encoder.smiles_to_graph(batch["smiles"][b][i])
                if graph is not None:
                    formulation_graphs.append(graph)
                    formulation_concs.append(batch["concentrations"][b][i])

        mol_graphs.append(formulation_graphs)

        if formulation_concs:
            total_conc = sum(formulation_concs)
            normalized_concs = [
                c / total_conc if total_conc > 0 else 0 for c in formulation_concs
            ]
            normalized_concs.extend([0] * (max_molecules - len(normalized_concs)))
        else:
            normalized_concs = [0] * max_molecules

        all_concentrations.append(normalized_concs)

    concentrations = torch.tensor(all_concentrations, dtype=torch.float)

    return mol_graphs, concentrations
