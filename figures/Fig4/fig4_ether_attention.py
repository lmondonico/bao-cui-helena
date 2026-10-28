"""
Self-contained analyzer for local self-attention patterns on ether moieties.
Shows local self-attention scores from the moiety encoder for ether groups.
"""

import os
import math
import colorsys
import warnings
from collections import defaultdict
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

CUSTOM_SMILES = [
    "COCOC",
    "CCOCOCC",
    "CCCOCOCCC",
    "CCCCOCOCCCC",
    "COCCOC",
    "COCC(C)OC",
    "CCOCCOCC",
    "CCOCC(C)OCC",
    "CC(C)OCCOC(C)C",
]
SHOW_MOIETIES = ["ether"]

FLUORINATED_ETHERS_SMILES = [
    "CCOCCOCC",
    "C(F)COCCOCC",
    "C(F)COCCOCC(F)",
    "C(F)COCCOCC(F)(F)",
    "FC(F)(F)COCCOCC",
    "FC(F)COCCOCC(F)(F)",
    "FC(F)COCCOCC(F)(F)(F)",
    "FC(F)(F)COCCOCC(F)(F)(F)",
]
FLUORINATED_MOIETIES = ["ether"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "helena_ce_trained_weights.pth")
OUTPUT_DIR = SCRIPT_DIR

# Architecture parameters
ARCHITECTURE = {
    "d_model": 128,
    "moiety_att_num_layers": 2,
    "moiety_att_num_heads": 4,
    "moiety_encoder_dropout": 0.15,
    "group_att_num_heads": 4,
    "group_att_num_layers": 2,
    "main_att_num_heads": 4,
    "main_att_dropout": 0.1,
    "final_dropout": 0.15,
    "head_dim": 512,
    "ce_head_hidden_dim": 64,
}

MAX_SOLVENTS = 5
MAX_SALTS = 3
MAX_MOLECULES = MAX_SOLVENTS + MAX_SALTS

# Moiety dictionary
MOIETIES = {
    "methyl": "[CX4H3]",
    "secondary_carbon": "[#6X4H2]",
    "tertiary_carbon": "[#6X4H1]",
    "quaternary_carbon": "[#6X4H0]",
    "alkene": "[CX3]=[CX3]",
    "benzene": "c1ccccc1",
    "naphthalene": "c12ccccc1cccc2",
    "ether": "[OX2]",
    "carbonyl": "[C](=O)",
    "carboxylate": "[CX3](=O)[O-]",
    "carbonate_ester": "[#6][OX2]C(=O)[OX2][#6]",
    "trivalent_nitrogen": "[NX3]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "nitrile": "[NX1]#[CX2]",
    "disulfide": "[SX2;!$([S-])][SX2;!$([S-])]",
    "trisulfide": "[SX2;!$([S-])][SX2;!$([S-])][SX2;!$([S-])]",
    "sulfinyl": "[SX3](=O)",
    "sulfonyl": "[SX4](=O)(=O)",
    "phosphoryl": "[P](=O)",
    "cyclotriphosphazene": "[N]1=P[N]=P[N]=P1",
    "trigonal_borate": "[BX3](O)(O)O",
    "silyl": "[SiX4]",
    "fluoro": "[FX1]",
    "chloro": "[ClX1]",
    "perchlorate_chlorine": "[ClX4]",
    "bromo": "[BrX1]",
    "iodo": "[IX1]",
    "ammonium_ion": "[NX4+]",
    "arsenide_ion": "[As-]",
    "boride_ion": "[B-]",
    "lithium_cation": "[Li+]",
    "nitrogen_anion": "[N-]",
    "oxygen_anion": "[O-]",
    "phosphorus_anion": "[P-]",
    "sulfur_anion": "[S-]",
    "nitrate_ion": "[N+](=O)([O-])[O-]",
}


# =============================================================================
# Tokenizer
# =============================================================================

class Tokenizer:
    def __init__(self):
        self.fg_dict = MOIETIES
        self.moiety_key_map = {name: idx + 1 for idx, name in enumerate(self.fg_dict.keys())}
        self.key_moiety_map = {idx: name for name, idx in self.moiety_key_map.items()}
        self.next_key = max(self.moiety_key_map.values()) + 1
        self.global_color_map = {}

    def detect_MOIETIES(self, mol, fg_dict):
        fg_matches = []
        for fg_name, smarts in fg_dict.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            pattern_atom_labels = [atom.GetSmarts() for atom in pattern.GetAtoms()]
            matches = mol.GetSubstructMatches(pattern, uniquify=False)
            for match in matches:
                included_atoms = []
                for pat_idx, mol_idx in enumerate(match):
                    label = pattern_atom_labels[pat_idx]
                    if label != "[#6]":
                        included_atoms.append(mol_idx)
                if included_atoms:
                    fg_matches.append((fg_name, set(included_atoms), "moiety"))

        ring_matches = self.detect_rings(mol)
        all_matches = fg_matches + ring_matches
        non_overlapping_matches = self.filter_overlapping_matches(all_matches)
        return non_overlapping_matches

    def detect_rings(self, mol):
        ring_matches = []
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        ring_systems = self.group_rings(rings)
        for ring_atoms_list in ring_systems:
            ring_sizes = [len(ring) for ring in ring_atoms_list]
            ring_sizes_str = "_&_".join(str(size) for size in sorted(ring_sizes))
            ring_name = f"ring_{ring_sizes_str}"
            atom_indices = set()
            for ring in ring_atoms_list:
                atom_indices.update(ring)
            ring_matches.append((ring_name, atom_indices, "ring"))
            if ring_name not in self.moiety_key_map:
                self.moiety_key_map[ring_name] = self.next_key
                self.key_moiety_map[self.next_key] = ring_name
                self.global_color_map[self.next_key] = self.generate_single_color(self.next_key)
                self.next_key += 1
        return ring_matches

    def group_rings(self, rings):
        if not rings:
            return []
        ring_sets = [set(ring) for ring in rings]
        ring_systems = []
        while ring_sets:
            first, *rest = ring_sets
            first = set(first)
            merged = False
            rest2 = []
            for r in rest:
                if first & r:
                    first |= r
                    merged = True
                else:
                    rest2.append(r)
            if merged:
                ring_sets = [first] + rest2
            else:
                ring_systems.append([tuple(first)])
                ring_sets = rest2
        final_ring_systems = []
        for system in ring_systems:
            if not system:
                continue
            system_atoms = system.pop()
            individual_rings = [ring for ring in rings if set(ring).issubset(system_atoms)]
            final_ring_systems.append(individual_rings)
        return final_ring_systems

    def filter_overlapping_matches(self, matches):
        matches.sort(key=lambda x: (x[2] == "ring", -len(x[1])))
        non_overlapping = []
        used_atoms = set()
        for fg_name, atom_indices, moiety_type in matches:
            if not atom_indices & used_atoms:
                non_overlapping.append((fg_name, atom_indices))
                used_atoms.update(atom_indices)
        return non_overlapping

    def generate_single_color(self, key):
        hue = (key * 0.618033988749895) % 1
        saturation = 0.6 + 0.4 * (key % 2)
        value = 0.95
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def get_moieties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        matches = self.detect_MOIETIES(mol, self.fg_dict)
        ring_info = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)
        is_in_ring = [1 if atom_indices & ring_atoms else 0 for _, atom_indices in matches]
        clique_atom_ids = [sorted(list(atom_indices)) for _, atom_indices in matches]
        moiety_ids = [self.moiety_key_map.get(fg_name, -1) for fg_name, _ in matches]
        atom_to_moiety = {}
        for moiety_idx, (fg_name, atom_indices) in enumerate(matches):
            for atom in atom_indices:
                atom_to_moiety[atom] = moiety_idx
        edges = set()
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            moiety1 = atom_to_moiety.get(begin)
            moiety2 = atom_to_moiety.get(end)
            if moiety1 is not None and moiety2 is not None and moiety1 != moiety2:
                edges.add(tuple(sorted((begin, end))))
        edges = sorted(list(edges))
        return clique_atom_ids, edges, moiety_ids, matches, is_in_ring


# =============================================================================
# Moiety feature functions
# =============================================================================

def map_atom_to_moiety(clique_atom_ids):
    atom_to_moiety = {}
    for moiety_index, atoms_in_moiety in enumerate(clique_atom_ids):
        for atom_idx in atoms_in_moiety:
            atom_to_moiety[atom_idx] = moiety_index
    return atom_to_moiety


def build_moiety_edges(edges, atom_to_moiety):
    moiety_edges = set()
    for u, v in edges:
        moiety_u = atom_to_moiety.get(u)
        moiety_v = atom_to_moiety.get(v)
        if moiety_u is not None and moiety_v is not None and moiety_u != moiety_v:
            moiety_edges.add((moiety_u, moiety_v))
    return moiety_edges


def get_adj_matrix(node_list, edges):
    adjoin_matrix = np.eye(len(node_list))
    for u, v in edges:
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return adjoin_matrix


def get_dist_matrix(node_list, edges):
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.zeros((len(node_list), len(node_list)))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row, col] = 0
    graph_nodes = sorted(make_graph.nodes.keys())
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    for node in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[node].items(), key=lambda x: x[0]))
        temp_node_dist_dict = {
            i: node_relative_distance.get(i) if node_relative_distance.get(i) is not None else 1e9
            for i in graph_nodes
        }
        temp_node_dist_list = list(temp_node_dist_dict.values())
        dist_matrix[node][graph_nodes] = temp_node_dist_list
    return dist_matrix.astype(np.float32)


# =============================================================================
# Model components
# =============================================================================

def create_padding_mask(batch_data):
    return (batch_data == 0).float().unsqueeze(1).unsqueeze(2)


def rescale_distance_matrix(w):
    c = torch.tensor(1.0, dtype=torch.float32, device=w.device)
    return (c + torch.exp(c)) / (c + torch.exp(c - w))


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    dk = q.size(-1)
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    if dist_matrix is not None:
        matmul_qk = F.relu(matmul_qk)
        dist_matrix = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = (matmul_qk * dist_matrix) / math.sqrt(dk)
    elif adjoin_matrix is not None:
        scaled_attention_logits = (matmul_qk * adjoin_matrix) / math.sqrt(dk)
    else:
        scaled_attention_logits = matmul_qk / math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + mask * -1e9
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)
        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super().__init__()
        self.multi_head_att_1 = MultiHeadAttention(d_model // 2, num_heads)
        self.multi_head_att_2 = MultiHeadAttention(d_model // 2, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff), nn.GELU(), nn.Linear(dff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x_l, attn_weights_local = self.multi_head_att_1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )
        x_g, attn_weights_global = self.multi_head_att_2(
            x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix
        )
        attn_output = torch.cat([x_l, x_g], dim=-1)
        out1 = self.layer_norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layer_norm2(out1 + self.dropout2(ffn_output))
        return out2, attn_weights_local, attn_weights_global


class MoietyEncoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        padding_mask = create_padding_mask(x)
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.moiety_linear(self.dropout(x))
        attn_weights_local_list = []
        attn_weights_global_list = []
        for layer in self.encoder_layers:
            x, attn_local, attn_global = layer(x, padding_mask, adjoin_matrix, dist_matrix)
            attn_weights_local_list.append(attn_local)
            attn_weights_global_list.append(attn_global)
        return x, attn_weights_local_list, attn_weights_global_list, padding_mask


class MolarRatioEncoder(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.d_model = d_model

    def forward(self, molar_ratios):
        zero_mask = molar_ratios == 0
        masked_ratios = molar_ratios.masked_fill(zero_mask, 0.0)
        ratio_sum = masked_ratios.sum(dim=1, keepdim=True).clamp(min=1e-6)
        mr_weights = (masked_ratios / ratio_sum).unsqueeze(-1)
        return mr_weights.expand(-1, -1, self.d_model)


class FormulationAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_mask = ~mask
        empty_samples = ~mask.any(dim=1)
        if empty_samples.any():
            attn_mask = attn_mask.clone()
            attn_mask[empty_samples, 0] = False
        attended_x, attention_weights = self.attention(
            query=x, key=x, value=x, key_padding_mask=attn_mask
        )
        x = self.layer_norm(x + self.dropout(attended_x))
        x = torch.nan_to_num(x * mask.unsqueeze(-1).float(), nan=0.0)
        return x, attention_weights


class GlobalPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        masked_x = x * mask.unsqueeze(-1)
        return torch.nan_to_num(masked_x.sum(dim=1), nan=0.0)


class SolventGroup(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, scaled_mol_embeddings, mask):
        attn_mask = ~mask
        x = scaled_mol_embeddings
        empty_samples = ~mask.any(dim=1)
        if empty_samples.any():
            attn_mask = attn_mask.clone()
            attn_mask[empty_samples, 0] = False
        for attn, norm in zip(self.layers, self.norms):
            out, weights = attn(x, x, x, key_padding_mask=attn_mask)
            x = norm(out + x)
        valid_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = (x * mask.unsqueeze(-1).float()).sum(dim=1) / valid_count
        return torch.nan_to_num(pooled, nan=0.0), weights


class SaltGroup(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mol_embeddings, mask):
        masked_embeddings = mol_embeddings * mask.unsqueeze(-1).float()
        valid_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = masked_embeddings.sum(dim=1) / valid_count
        return torch.nan_to_num(pooled, nan=0.0)


class HELENA_CE(nn.Module):
    def __init__(
        self,
        d_model,
        moiety_att_num_layers,
        moiety_att_num_heads,
        moiety_encoder_dropout,
        group_att_num_heads,
        group_att_num_layers,
        main_att_num_heads,
        main_att_dropout,
        final_dropout,
        head_dim,
        ce_head_hidden_dim,
        input_vocab_size,
        max_num_molecules,
        task_type="regression",
        num_classes=None,
    ):
        super().__init__()
        self.max_num_molecules = max_num_molecules
        self.task_type = task_type
        self.num_classes = num_classes
        self.encoder_moiety = MoietyEncoder(
            input_vocab_size=input_vocab_size,
            d_model=d_model,
            dff=d_model,
            num_layers=moiety_att_num_layers,
            num_heads=moiety_att_num_heads,
            rate=moiety_encoder_dropout,
        )
        self.transf_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.transf_norm = nn.LayerNorm(d_model)
        self.molar_ratio_encoder = MolarRatioEncoder(d_model)
        self.solvent_group = SolventGroup(
            d_model, group_att_num_heads, group_att_num_layers, main_att_dropout
        )
        self.salt_group = SaltGroup()
        self.molecular_attention = FormulationAttention(
            d_model=d_model, num_heads=main_att_num_heads, dropout=main_att_dropout
        )
        self.global_pooling = GlobalPooling()
        self.formulation_norm = nn.LayerNorm(d_model)
        ce_input_dim = d_model + 2
        output_dim = 1 if task_type == "regression" else num_classes
        self.ce_head = nn.Sequential(
            nn.Linear(ce_input_dim, ce_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(final_dropout),
            nn.Linear(ce_head_hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        num_residual_layers = 2 * self.encoder_moiety.num_layers + len(self.solvent_group.layers) + 2
        residual_scale = 1.0 / math.sqrt(2 * num_residual_layers)
        residual_output_keys = ["dense", "ffn.2", "out_proj", "transf_mlp.2"]
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name == "ce_head.3":
                    nn.init.trunc_normal_(module.weight, std=0.01)
                elif "ce_head" in name:
                    nn.init.trunc_normal_(module.weight, std=0.1)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                if any(key in name for key in residual_output_keys):
                    with torch.no_grad():
                        module.weight.mul_(residual_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


# =============================================================================
# Dataset processing
# =============================================================================

class FeatureEngineering:
    def __init__(self, data_set, tokenizer, map_dict, label_field, molar_ratio_noise):
        self.data_set = data_set.copy()
        self.label_field = label_field
        if label_field is not None:
            self.data_set[label_field] = self.data_set[label_field].astype(float)
        self.tokenizer = tokenizer
        self.pad_value = 0
        self.map_dict = map_dict
        self.max_num_solvent_molecules = MAX_SOLVENTS
        self.max_num_salt_molecules = MAX_SALTS
        self.max_num_molecules = MAX_MOLECULES
        self.molar_ratio_noise = molar_ratio_noise
        if "Method" in self.data_set.columns:
            self.data_set["Method"] = self.data_set["Method"].fillna("Aurbach")
            self.data_set["method_numeric"] = self.data_set["Method"].map({"Aurbach": 0, "Cycle": 1})
        else:
            self.data_set["method_numeric"] = 0
        if "Current" in self.data_set.columns:
            self.data_set["Current"] = self.data_set["Current"].fillna(0.5)
            self.data_set["current_numeric"] = self.data_set["Current"].astype(float)
        else:
            self.data_set["current_numeric"] = 0.5
        self.data_list = []
        self._process_data()

    def _process_data(self):
        for idx, row in self.data_set.iterrows():
            smiles_list = []
            molar_ratios = []
            molecule_types = []
            for i in range(1, self.max_num_solvent_molecules + 1):
                mol_col = f"Solvent_{i}"
                if mol_col in row and pd.notna(row[mol_col]):
                    parts = str(row[mol_col]).split(":")
                    if len(parts) == 2:
                        smiles = parts[0].strip()
                        molar_ratio = float(parts[1].strip())
                        smiles_list.append(smiles)
                        molar_ratios.append(molar_ratio)
                        molecule_types.append(1)
            for i in range(1, self.max_num_salt_molecules + 1):
                mol_col = f"Salt_{i}"
                if mol_col in row and pd.notna(row[mol_col]):
                    parts = str(row[mol_col]).split(":")
                    if len(parts) == 2:
                        smiles = parts[0].strip()
                        molar_ratio = float(parts[1].strip())
                        smiles_list.append(smiles)
                        molar_ratios.append(molar_ratio)
                        molecule_types.append(2)
            num_molecules = len(smiles_list)
            padding_length = self.max_num_molecules - num_molecules
            if padding_length > 0:
                smiles_list.extend([None] * padding_length)
                molar_ratios.extend([0.0] * padding_length)
                molecule_types.extend([0] * padding_length)
            label = row[self.label_field] if self.label_field is not None else 0.0
            row_id = row["Row_ID"]
            method = row["method_numeric"]
            current = row["current_numeric"]
            self.data_list.append((smiles_list, label, molar_ratios, molecule_types, row_id, method, current))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        smiles_list, label, molar_ratios, molecule_types, row_id, method, current = self.data_list[idx]
        sample = self.numerical_seq(smiles_list, label, molar_ratios, molecule_types)
        sample["row_id"] = row_id
        sample["method"] = torch.tensor(method, dtype=torch.long)
        sample["current"] = torch.tensor(current, dtype=torch.float32)
        return sample

    def numerical_seq(self, smiles_list, label, molar_ratios, molecule_types):
        sample = {}
        num_mols = sum(1 for smiles in smiles_list if smiles is not None)
        sample["num_mols"] = num_mols
        for i in range(1, self.max_num_molecules + 1):
            if i <= num_mols and smiles_list[i - 1] is not None:
                smiles = smiles_list[i - 1]
                moiety_ids = self.map_dict[smiles]["moiety_ids"]
                dist_matrix = self.map_dict[smiles]["dist_matrix"]
                adjoin_matrix = self.map_dict[smiles]["adj_matrix"]
                sample[f"x{i}"] = np.array(moiety_ids, dtype=np.int64)
                sample[f"adjoin_matrix{i}"] = adjoin_matrix
                sample[f"dist_matrix{i}"] = dist_matrix
                sample[f"mr{i}"] = float(molar_ratios[i - 1])
                sample[f"mol_type{i}"] = int(molecule_types[i - 1])
            else:
                sample[f"x{i}"] = np.array([self.pad_value], dtype=np.int64)
                sample[f"adjoin_matrix{i}"] = np.array([[-1e9]], dtype=np.float32)
                sample[f"dist_matrix{i}"] = np.array([[-1e9]], dtype=np.float32)
                sample[f"mr{i}"] = 0
                sample[f"mol_type{i}"] = 0
        sample["y"] = np.array([label], dtype=np.float32)
        return sample


class CollateFn:
    def __init__(self, max_num_mols):
        self.max_num_mols = max_num_mols

    def __call__(self, batch):
        max_num_mols = self.max_num_mols
        collated_batch = {"num_mols": max_num_mols}
        for i in range(1, max_num_mols + 1):
            collated_batch[f"x{i}"] = []
            collated_batch[f"adjoin_matrix{i}"] = []
            collated_batch[f"dist_matrix{i}"] = []
            collated_batch[f"mr{i}"] = []
            collated_batch[f"mol_type{i}"] = []
        y_list = []
        row_id_list = []
        method_list = []
        current_list = []
        for sample in batch:
            y_list.append(torch.as_tensor(sample["y"], dtype=torch.float32))
            row_id_list.append(torch.as_tensor(sample["row_id"], dtype=torch.long))
            method_list.append(torch.as_tensor(sample["method"], dtype=torch.long))
            current_list.append(torch.as_tensor(sample["current"], dtype=torch.float32))
            for i in range(1, max_num_mols + 1):
                if i <= sample["num_mols"]:
                    collated_batch[f"x{i}"].append(torch.as_tensor(sample[f"x{i}"], dtype=torch.int64))
                    collated_batch[f"adjoin_matrix{i}"].append(torch.as_tensor(sample[f"adjoin_matrix{i}"], dtype=torch.float32))
                    collated_batch[f"dist_matrix{i}"].append(torch.as_tensor(sample[f"dist_matrix{i}"], dtype=torch.float32))
                    collated_batch[f"mr{i}"].append(torch.as_tensor(sample[f"mr{i}"], dtype=torch.float32).unsqueeze(0))
                    collated_batch[f"mol_type{i}"].append(torch.as_tensor(sample[f"mol_type{i}"], dtype=torch.long).unsqueeze(0))
                else:
                    collated_batch[f"x{i}"].append(torch.as_tensor([0], dtype=torch.int64))
                    collated_batch[f"adjoin_matrix{i}"].append(torch.as_tensor([[-1e9]], dtype=torch.float32))
                    collated_batch[f"dist_matrix{i}"].append(torch.as_tensor([[-1e9]], dtype=torch.float32))
                    collated_batch[f"mr{i}"].append(torch.as_tensor([0.0], dtype=torch.float32))
                    collated_batch[f"mol_type{i}"].append(torch.as_tensor([0], dtype=torch.long))
        for i in range(1, max_num_mols + 1):
            collated_batch[f"x{i}"] = self.pad_sequence_list(collated_batch[f"x{i}"], padding_value=0)
            collated_batch[f"adjoin_matrix{i}"] = self.pad_matrix_list(collated_batch[f"adjoin_matrix{i}"], padding_value=-1e9)
            collated_batch[f"dist_matrix{i}"] = self.pad_matrix_list(collated_batch[f"dist_matrix{i}"], padding_value=-1e9)
            collated_batch[f"mr{i}"] = torch.cat(collated_batch[f"mr{i}"], dim=0)
            collated_batch[f"mol_type{i}"] = torch.cat(collated_batch[f"mol_type{i}"], dim=0)
        collated_batch["y"] = torch.cat(y_list, dim=0)
        collated_batch["row_id"] = torch.tensor(row_id_list)
        collated_batch["method"] = torch.stack(method_list)
        collated_batch["current"] = torch.stack(current_list)
        return collated_batch

    def pad_sequence_list(self, sequence_list, padding_value=0):
        if not sequence_list:
            return torch.empty(0)
        return torch.nn.utils.rnn.pad_sequence(sequence_list, batch_first=True, padding_value=padding_value)

    def pad_matrix_list(self, matrix_list, padding_value=0):
        if not matrix_list:
            return torch.empty(0)
        max_height = max(m.shape[0] for m in matrix_list)
        max_width = max(m.shape[1] for m in matrix_list)
        padded_matrices = []
        for m in matrix_list:
            height, width = m.shape
            pad_height = max_height - height
            pad_width = max_width - width
            m_tensor = torch.as_tensor(m)
            m_padded = torch.nn.functional.pad(m_tensor, (0, pad_width, 0, pad_height), value=padding_value)
            padded_matrices.append(m_padded)
        return torch.stack(padded_matrices)


# =============================================================================
# Attention utilities
# =============================================================================

class MoietyEncoderAttentionUtils:
    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path or MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer()
        self.model, self.map_dict, self.moiety_key_map, self.key_moiety_map = self.load_model()
        self.tokenizer.moiety_key_map = self.moiety_key_map
        self.tokenizer.key_moiety_map = self.key_moiety_map

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        moiety_key_map = checkpoint.get("moiety_key_map", {})
        key_moiety_map = checkpoint.get("key_moiety_map", {})
        vocab_size = checkpoint.get("vocab_size", max(moiety_key_map.values()) + 1 if moiety_key_map else 50)
        task_type = checkpoint.get("task_type", "regression")
        num_classes = checkpoint.get("num_classes", None)
        map_dict = {}
        model = HELENA_CE(
            **ARCHITECTURE,
            input_vocab_size=vocab_size,
            max_num_molecules=MAX_MOLECULES,
            task_type=task_type,
            num_classes=num_classes,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        print(f"Model loaded from: {self.checkpoint_path}")
        print(f"Task type: {task_type}, Vocab size: {vocab_size}")
        return model, map_dict, moiety_key_map, key_moiety_map

    def _ensure_molecule_in_map_dict(self, smiles):
        if smiles in self.map_dict:
            return True
        try:
            result = self.tokenizer.get_moieties(smiles)
            if result is None:
                return False
            clique_atom_ids, edges, moiety_ids, _, is_in_ring = result
            atom_to_moiety = map_atom_to_moiety(clique_atom_ids)
            moiety_edges = build_moiety_edges(edges, atom_to_moiety)
            node_list = list(range(len(clique_atom_ids)))
            moiety_edges_list = list(moiety_edges)
            dist_matrix = get_dist_matrix(node_list, moiety_edges_list)
            adj_matrix = get_adj_matrix(node_list, moiety_edges_list)
            self.map_dict[smiles] = {
                "adj_matrix": adj_matrix,
                "dist_matrix": dist_matrix,
                "clique_atom_ids": clique_atom_ids,
                "edges": edges,
                "moiety_ids": moiety_ids,
                "is_in_ring": is_in_ring,
            }
            return True
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return False

    def get_attention_weights(self, mol_smiles, molar_ratio=1.0):
        try:
            result = self.tokenizer.get_moieties(mol_smiles)
            if result is None:
                return None, None, None, None
            clique_atom_ids, edges, moiety_ids, *_ = result
            if not self._ensure_molecule_in_map_dict(mol_smiles):
                return None, None, None, None
            temp_data = pd.DataFrame({
                "Solvent_1": [f"{mol_smiles}:{molar_ratio}"],
                "Salt_1": [None],
                "LCE": [0.0],
                "Row_ID": [0],
                "Method": ["Cycle"],
                "Current": [0.5],
            })
            dataset = FeatureEngineering(
                temp_data,
                self.tokenizer,
                self.map_dict,
                label_field="LCE",
                molar_ratio_noise=0.0,
            )
            collate_fn = CollateFn(max_num_mols=MAX_MOLECULES)
            batch = collate_fn([dataset[0]])
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                x1 = batch["x1"]
                adj_matrix = batch["adjoin_matrix1"]
                dist_matrix = batch["dist_matrix1"]
                _, local_attn, global_attn, _ = self.model.encoder_moiety(
                    x1, adjoin_matrix=adj_matrix, dist_matrix=dist_matrix
                )
                local_attn = torch.stack(local_attn).mean(0).squeeze(0).mean(0)
                global_attn = torch.stack(global_attn).mean(0).squeeze(0).mean(0)
                return (
                    local_attn.cpu().numpy(),
                    global_attn.cpu().numpy(),
                    moiety_ids,
                    clique_atom_ids,
                )
        except Exception as e:
            print(f"Error processing molecule {mol_smiles}: {e}")
            return None, None, None, None


# =============================================================================
# Analyzer
# =============================================================================

class SimpleAttentionAnalyzer:
    def __init__(self):
        self.heatmap_analyzer = MoietyEncoderAttentionUtils()
        # Custom colormap from yellow to green (for original ethers)
        from matplotlib.colors import LinearSegmentedColormap
        self.local_cmap = LinearSegmentedColormap.from_list(
            "YellowGreen", ["#FFFF00", "#00AA00"]
        )
        # Original YlOrRd colormap for fluorinated ethers
        self.fluoro_cmap = plt.cm.YlOrRd

    def analyze_molecules_vertically(self, smiles_list, output_dir, show_moieties=None):
        print(f"\nProcessing molecules (highlighted moieties: {', '.join(show_moieties)})...")
        molecules_data = []
        all_self_attention_scores = []

        for smiles in smiles_list:
            try:
                local_attn, global_attn, moiety_ids, clique_atom_ids = (
                    self.heatmap_analyzer.get_attention_weights(smiles, 1.0)
                )
                if local_attn is None:
                    print(f"Could not process {smiles}, skipping")
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                if mol.GetNumConformers() == 0:
                    rdDepictor.Compute2DCoords(mol)
                moiety_names = [
                    self.heatmap_analyzer.tokenizer.key_moiety_map.get(mid)
                    for mid in moiety_ids
                ]
                moiety_type_scores = defaultdict(list)
                moiety_type_indices = defaultdict(list)
                for idx, name in enumerate(moiety_names):
                    if show_moieties is None or name in show_moieties:
                        self_attn = float(local_attn[idx, idx])
                        moiety_type_scores[name].append(self_attn)
                        moiety_type_indices[name].append(idx)
                moiety_type_avg = {name: np.mean(scores) for name, scores in moiety_type_scores.items()}
                self_attn_scores = {}
                for name, indices in moiety_type_indices.items():
                    avg_score = moiety_type_avg[name]
                    for idx in indices:
                        self_attn_scores[idx] = avg_score
                    all_self_attention_scores.append(avg_score)
                molecules_data.append({
                    "smiles": smiles,
                    "mol": mol,
                    "moiety_ids": moiety_ids,
                    "moiety_names": moiety_names,
                    "clique_atom_ids": clique_atom_ids,
                    "self_attn_scores": self_attn_scores,
                })
            except Exception as e:
                print(f"Error processing {smiles}: {str(e)}")
                continue

        if not molecules_data or not all_self_attention_scores:
            print("No valid molecules to display")
            return

        global_min = min(all_self_attention_scores)
        global_max = max(all_self_attention_scores)
        score_range = global_max - global_min if global_max > global_min else 1.0

        n_mols = len(molecules_data)
        col1_mols = min(4, n_mols)
        col2_mols = min(5, max(0, n_mols - col1_mols))
        max_rows = max(col1_mols, col2_mols)

        fig, axes = plt.subplots(max_rows, 2, figsize=(8, 3.5 * max_rows))
        axes = np.atleast_2d(axes)
        axes[0, 0].set_title("Local Self-Attention (Col 1)", fontsize=12)
        axes[0, 1].set_title("Local Self-Attention (Col 2)", fontsize=12)

        for idx, data in enumerate(molecules_data):
            if idx < col1_mols:
                row = idx
                col = 0
            else:
                row = idx - col1_mols
                col = 1

            mol = data["mol"]
            clique_atom_ids = data["clique_atom_ids"]
            self_attn_scores = data["self_attn_scores"]

            atom_colors = {}
            highlight_atoms = set()
            for moiety_idx, self_attn in self_attn_scores.items():
                normalized = (self_attn - global_min) / score_range
                rgb_color = self.local_cmap(normalized)[:3]
                for atom_idx in clique_atom_ids[moiety_idx]:
                    atom_colors[atom_idx] = rgb_color
                    highlight_atoms.add(atom_idx)

            highlight_bonds = []
            bond_colors = {}
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_idx = bond.GetIdx()
                if begin_idx in highlight_atoms or end_idx in highlight_atoms:
                    highlight_bonds.append(bond_idx)
                    if begin_idx in atom_colors:
                        bond_colors[bond_idx] = atom_colors[begin_idx]
                    elif end_idx in atom_colors:
                        bond_colors[bond_idx] = atom_colors[end_idx]

            drawer = rdMolDraw2D.MolDraw2DCairo(600, 300)
            drawer.drawOptions().bondLineWidth = 2
            drawer.drawOptions().continuousHighlight = True
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(atom_colors.keys()),
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()
            img = Image.open(BytesIO(drawer.GetDrawingText()))
            ax = axes[row, col]
            ax.imshow(np.array(img))
            ax.set_title(data["smiles"], fontsize=9)
            ax.axis("off")

        for row in range(max_rows):
            if row >= col1_mols:
                axes[row, 0].axis("off")
                axes[row, 0].set_visible(False)
            if row >= col2_mols:
                axes[row, 1].axis("off")
                axes[row, 1].set_visible(False)

        cbar_ax = fig.add_axes([0.17, 0.05, 0.7, 0.02])
        norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=self.local_cmap),
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.set_label(f"Local Self-Attention (min={global_min:.3f}, max={global_max:.3f})")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        filename = "fig4_ether_attention.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Results saved to {output_dir}/{filename}")

    def analyze_fluorinated_ethers(self, smiles_list, output_dir, show_moieties=None):
        """Analyze fluorinated ethers with YlOrRd colormap."""
        print(f"\nProcessing fluorinated ethers (highlighted moieties: {', '.join(show_moieties)})...")
        molecules_data = []
        all_self_attention_scores = []

        for smiles in smiles_list:
            try:
                local_attn, global_attn, moiety_ids, clique_atom_ids = (
                    self.heatmap_analyzer.get_attention_weights(smiles, 1.0)
                )
                if local_attn is None:
                    print(f"Could not process {smiles}, skipping")
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                if mol.GetNumConformers() == 0:
                    rdDepictor.Compute2DCoords(mol)
                moiety_names = [
                    self.heatmap_analyzer.tokenizer.key_moiety_map.get(mid)
                    for mid in moiety_ids
                ]
                moiety_type_scores = defaultdict(list)
                moiety_type_indices = defaultdict(list)
                for idx, name in enumerate(moiety_names):
                    if show_moieties is None or name in show_moieties:
                        self_attn = float(local_attn[idx, idx])
                        moiety_type_scores[name].append(self_attn)
                        moiety_type_indices[name].append(idx)
                moiety_type_avg = {name: np.mean(scores) for name, scores in moiety_type_scores.items()}
                self_attn_scores = {}
                for name, indices in moiety_type_indices.items():
                    avg_score = moiety_type_avg[name]
                    for idx in indices:
                        self_attn_scores[idx] = avg_score
                    all_self_attention_scores.append(avg_score)
                molecules_data.append({
                    "smiles": smiles,
                    "mol": mol,
                    "moiety_ids": moiety_ids,
                    "moiety_names": moiety_names,
                    "clique_atom_ids": clique_atom_ids,
                    "self_attn_scores": self_attn_scores,
                })
            except Exception as e:
                print(f"Error processing {smiles}: {str(e)}")
                continue

        if not molecules_data or not all_self_attention_scores:
            print("No valid molecules to display")
            return

        global_min = min(all_self_attention_scores)
        global_max = max(all_self_attention_scores)
        score_range = global_max - global_min if global_max > global_min else 1.0

        n_mols = len(molecules_data)
        col1_mols = min(4, n_mols)
        col2_mols = min(4, max(0, n_mols - col1_mols))
        max_rows = max(col1_mols, col2_mols)

        fig, axes = plt.subplots(max_rows, 2, figsize=(8, 3.5 * max_rows))
        axes = np.atleast_2d(axes)
        axes[0, 0].set_title("Local Self-Attention (Col 1)", fontsize=12)
        axes[0, 1].set_title("Local Self-Attention (Col 2)", fontsize=12)

        for idx, data in enumerate(molecules_data):
            if idx < col1_mols:
                row = idx
                col = 0
            else:
                row = idx - col1_mols
                col = 1

            mol = data["mol"]
            clique_atom_ids = data["clique_atom_ids"]
            self_attn_scores = data["self_attn_scores"]

            atom_colors = {}
            highlight_atoms = set()
            for moiety_idx, self_attn in self_attn_scores.items():
                normalized = (self_attn - global_min) / score_range
                rgb_color = self.fluoro_cmap(normalized)[:3]  # Use YlOrRd colormap
                for atom_idx in clique_atom_ids[moiety_idx]:
                    atom_colors[atom_idx] = rgb_color
                    highlight_atoms.add(atom_idx)

            highlight_bonds = []
            bond_colors = {}
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_idx = bond.GetIdx()
                if begin_idx in highlight_atoms or end_idx in highlight_atoms:
                    highlight_bonds.append(bond_idx)
                    if begin_idx in atom_colors:
                        bond_colors[bond_idx] = atom_colors[begin_idx]
                    elif end_idx in atom_colors:
                        bond_colors[bond_idx] = atom_colors[end_idx]

            drawer = rdMolDraw2D.MolDraw2DCairo(600, 300)
            drawer.drawOptions().bondLineWidth = 2
            drawer.drawOptions().continuousHighlight = True
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(atom_colors.keys()),
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()
            img = Image.open(BytesIO(drawer.GetDrawingText()))
            ax = axes[row, col]
            ax.imshow(np.array(img))
            ax.set_title(data["smiles"], fontsize=9)
            ax.axis("off")

        for row in range(max_rows):
            if row >= col1_mols:
                axes[row, 0].axis("off")
                axes[row, 0].set_visible(False)
            if row >= col2_mols:
                axes[row, 1].axis("off")
                axes[row, 1].set_visible(False)

        cbar_ax = fig.add_axes([0.17, 0.05, 0.7, 0.02])
        norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=self.fluoro_cmap),
            cax=cbar_ax,
            orientation="horizontal",
        )
        cbar.set_label(f"Local Self-Attention (min={global_min:.3f}, max={global_max:.3f})")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        filename = "fig4_ether_fluoro_attention.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Results saved to {output_dir}/{filename}")


def main():
    print()
    print("+" * 50)
    print("Ether Local Self-Attention Analysis")
    print("+" * 50)

    analyzer = SimpleAttentionAnalyzer()

    # Original ethers analysis
    print(f"\nAnalyzing {len(CUSTOM_SMILES)} custom SMILES molecules")
    print(f"Showing moieties: {SHOW_MOIETIES}")
    analyzer.analyze_molecules_vertically(
        CUSTOM_SMILES,
        OUTPUT_DIR,
        show_moieties=SHOW_MOIETIES,
    )

    # Fluorinated ethers analysis
    print(f"\nAnalyzing {len(FLUORINATED_ETHERS_SMILES)} fluorinated ether molecules")
    print(f"Showing moieties: {FLUORINATED_MOIETIES}")
    analyzer.analyze_fluorinated_ethers(
        FLUORINATED_ETHERS_SMILES,
        OUTPUT_DIR,
        show_moieties=FLUORINATED_MOIETIES,
    )

    print(f"\nAnalysis saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
