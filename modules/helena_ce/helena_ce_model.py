"""HELENA-CE Model for Coulombic Efficiency prediction."""

import torch
import torch.nn as nn
import math
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import importlib

helena_ce_components = importlib.import_module("modules.helena_ce.helena_ce_components")
MoietyEncoder = helena_ce_components.MoietyEncoder
MolarRatioEncoder = helena_ce_components.MolarRatioEncoder
FormulationAttention = helena_ce_components.FormulationAttention
GlobalPooling = helena_ce_components.GlobalPooling
SolventGroup = helena_ce_components.SolventGroup
SaltGroup = helena_ce_components.SaltGroup


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

        # Moiety encoder
        self.encoder_moiety = MoietyEncoder(
            input_vocab_size=input_vocab_size,
            d_model=d_model,
            dff=d_model,
            num_layers=moiety_att_num_layers,
            num_heads=moiety_att_num_heads,
            rate=moiety_encoder_dropout,
        )

        # Transform MLP after moiety encoding
        self.transf_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.transf_norm = nn.LayerNorm(d_model)

        # Molar ratio encoder
        self.molar_ratio_encoder = MolarRatioEncoder(d_model)

        # Group attention
        self.solvent_group = SolventGroup(
            d_model, group_att_num_heads, group_att_num_layers, main_att_dropout
        )
        self.salt_group = SaltGroup()

        # Formulation attention and pooling
        self.molecular_attention = FormulationAttention(
            d_model=d_model, num_heads=main_att_num_heads, dropout=main_att_dropout
        )
        self.global_pooling = GlobalPooling()

        # LayerNorm for formulation features
        self.formulation_norm = nn.LayerNorm(d_model)

        # Prediction head: concat(formulation, ln_current, method) -> output
        # formulation: d_model, ln_current: 1, method: 1
        ce_input_dim = d_model + 2
        output_dim = 1 if task_type == "regression" else num_classes
        self.ce_head = nn.Sequential(
            nn.Linear(ce_input_dim, ce_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(final_dropout),
            nn.Linear(ce_head_hidden_dim, output_dim),
        )

        self._init_weights()

    def set_output_bias(self, mean_target):
        """Set final layer bias for faster convergence."""
        if self.task_type == "regression":
            with torch.no_grad():
                self.ce_head[-1].bias.fill_(mean_target)

    def _init_weights(self):
        num_residual_layers = (
            2 * self.encoder_moiety.num_layers + len(self.solvent_group.layers) + 2
        )
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

    def forward(self, batch):
        device = next(self.parameters()).device
        mol_embeddings = []
        molar_ratios = []
        molecule_types = []

        # Encode each molecule
        for i in range(1, self.max_num_molecules + 1):
            moiety_inputs = batch[f"x{i}"].to(device)
            moiety_adj_inputs = batch[f"adjoin_matrix{i}"].to(device)
            moiety_dist_inputs = batch[f"dist_matrix{i}"].to(device)
            mr = batch[f"mr{i}"].to(device)
            mol_type = batch[f"mol_type{i}"].to(device)

            molar_ratios.append(mr)
            molecule_types.append(mol_type)

            mol_moiety, *_ = self.encoder_moiety(
                moiety_inputs,
                adjoin_matrix=moiety_adj_inputs,
                dist_matrix=moiety_dist_inputs,
            )

            transformed = self.transf_mlp(mol_moiety)
            mol_embedding = self.transf_norm(transformed + mol_moiety)
            mask = (moiety_inputs != 0).float().unsqueeze(-1)
            mol_embedding = (mol_embedding * mask).sum(dim=1)
            mol_embeddings.append(mol_embedding)

        mol_embeddings = torch.stack(mol_embeddings, dim=1)
        molar_ratios = torch.stack(molar_ratios, dim=1)
        molecule_types = torch.stack(molecule_types, dim=1)

        # Scale by molar ratios
        mr_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_mol_embeddings = mol_embeddings * mr_weights

        # Group processing
        solvent_mask = (molar_ratios > 0) & (molecule_types == 1)
        salt_mask = (molar_ratios > 0) & (molecule_types == 2)

        solvent_out, _ = self.solvent_group(scaled_mol_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_mol_embeddings, salt_mask)

        # Formulation attention
        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack(
            [solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1
        )
        attended_embeddings, _ = self.molecular_attention(combined, combined_mask)
        combined_output = self.global_pooling(attended_embeddings, combined_mask)

        # Normalize formulation features
        formulation_features = self.formulation_norm(combined_output)

        # Direct scalar features (no learned encoding)
        current = batch["current"].to(device)
        if current.dim() == 1:
            current = current.unsqueeze(-1)
        ln_current = torch.log(current + 1e-6)  # (batch_size, 1)

        method = batch["method"].to(device).float().unsqueeze(-1)  # (batch_size, 1)

        # Concatenate: formulation (d_model) + ln_current (1) + method (1)
        combined_features = torch.cat(
            [formulation_features, ln_current, method], dim=-1
        )
        ce_output = self.ce_head(combined_features)

        return {"ce": ce_output}
