"""HELENA-CE backbone and Arrhenius temperature head for viscosity prediction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.paths import DATA_DIR
from modules.helena_ce.helena_ce_components import (
    FormulationAttention,
    GlobalPooling,
    MoietyEncoder,
    MolarRatioEncoder,
    SaltGroup,
    SolventGroup,
)
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE

HEAD_MODE = "ce_temperature_arrhenius"
CE_TEMPERATURE_HEAD_MODES = (HEAD_MODE,)
PRESERVED_MODULE_NAMES = (
    "encoder_moiety",
    "transf_mlp",
    "transf_norm",
    "molar_ratio_encoder",
    "solvent_group",
    "salt_group",
    "molecular_attention",
    "global_pooling",
    "formulation_norm",
)
PRESERVED_STATE_PREFIXES = tuple(f"{name}." for name in PRESERVED_MODULE_NAMES)
CE_CHECKPOINT = Path(DATA_DIR) / "helena_ce_trained_weights.pth"


def sanitize_graph_matrices(
    x: torch.Tensor,
    adjacency: torch.Tensor,
    distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero only graph entries incident to padding tokens.

    The inherited local-attention implementation multiplies dot products by
    adjacency before applying the token mask. A ``-1e9`` padding value can
    therefore become a large positive logit when the dot product is
    negative. Zero is neutral for that multiplication; the existing token
    mask remains solely responsible for excluding padded keys.
    """
    if x.ndim != 2:
        raise ValueError("Flattened token IDs must have shape [batch, tokens]")
    expected = (x.shape[0], x.shape[1], x.shape[1])
    if adjacency.shape != expected or distance.shape != expected:
        raise ValueError("Graph matrices do not match flattened token IDs")
    padding = x.eq(0)
    padding_pairs = padding.unsqueeze(-1) | padding.unsqueeze(-2)
    return (
        adjacency.masked_fill(padding_pairs, 0.0),
        distance.masked_fill(padding_pairs, 0.0),
    )


class HELENACETemperatureArrhenius(nn.Module):
    """HELENA-CE backbone plus one Arrhenius temperature head.

    standardized log10 eta = a(f) + softplus(b(f)) * x + r(f, T), with f the
    128-d CE formulation vector, x the standardized 1000/T feature, a and b
    linear maps of f, and r a CE-head-sized MLP over [f, both temperature
    features] whose last layer starts at exactly zero.
    """

    def __init__(
        self,
        d_model: int,
        moiety_att_num_layers: int,
        moiety_att_num_heads: int,
        moiety_encoder_dropout: float,
        group_att_num_heads: int,
        group_att_num_layers: int,
        main_att_num_heads: int,
        main_att_dropout: float,
        final_dropout: float,
        head_dim: int,
        ce_head_hidden_dim: int,
        input_vocab_size: int,
        max_num_molecules: int = 8,
        head_mode: str = "ce_temperature_arrhenius",
    ) -> None:
        super().__init__()
        if head_mode not in CE_TEMPERATURE_HEAD_MODES:
            raise ValueError(f"Unsupported CE-temperature head mode: {head_mode!r}")
        self.max_num_molecules = int(max_num_molecules)
        self.d_model = int(d_model)
        self.head_mode = head_mode
        self.published_unused_head_dim = int(head_dim)
        # The HELENA-CE backbone, built exactly as in HELENA-CE.
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
            d_model=d_model,
            num_heads=main_att_num_heads,
            dropout=main_att_dropout,
        )
        self.global_pooling = GlobalPooling()
        self.formulation_norm = nn.LayerNorm(d_model)
        # The single temperature head.
        self.arrhenius_intercept = nn.Linear(d_model, 1)
        self.arrhenius_slope = nn.Linear(d_model, 1)
        self.arrhenius_residual = nn.Sequential(
            nn.Linear(d_model + 2, ce_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(final_dropout),
            nn.Linear(ce_head_hidden_dim, 1),
        )
        self._init_weights()
        residual_output = self.arrhenius_residual[-1]
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

    def _init_weights(self) -> None:
        """Initialize weights (the backbone is then overwritten from the CE checkpoint)."""
        num_residual_layers = (
            2 * self.encoder_moiety.num_layers + len(self.solvent_group.layers) + 2
        )
        residual_scale = 1.0 / math.sqrt(2 * num_residual_layers)
        residual_output_keys = ("dense", "ffn.2", "out_proj", "transf_mlp.2")
        outputs = {"arrhenius_intercept", "arrhenius_slope", "arrhenius_residual.3"}
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.01 if name in outputs else 0.02)
                if any(key in name for key in residual_output_keys):
                    with torch.no_grad():
                        module.weight.mul_(residual_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_output_bias(self, mean_standardized_log10_viscosity: float) -> None:
        """Set the prediction at centered covariates in standardized space."""
        with torch.no_grad():
            self.arrhenius_intercept.bias.fill_(mean_standardized_log10_viscosity)

    def encode_formulation(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """The HELENA-CE formulation vector (graph padding sanitized)."""
        x = batch["x"]
        adjacency = batch["adjacency"]
        distance = batch["distance"]
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, molecules, tokens]")
        batch_size, molecule_count, token_count = x.shape
        if molecule_count != self.max_num_molecules:
            raise ValueError(
                f"Expected {self.max_num_molecules} molecule slots, got {molecule_count}"
            )
        expected_graph = (batch_size, molecule_count, token_count, token_count)
        if adjacency.shape != expected_graph or distance.shape != expected_graph:
            raise ValueError("Molecular graph tensors have incompatible shapes")
        x_flat = x.reshape(batch_size * molecule_count, token_count)
        clean_adjacency, clean_distance = sanitize_graph_matrices(
            x_flat,
            adjacency.reshape(batch_size * molecule_count, token_count, token_count),
            distance.reshape(batch_size * molecule_count, token_count, token_count),
        )
        moiety_embeddings, _, _, _ = self.encoder_moiety(
            x_flat, adjoin_matrix=clean_adjacency, dist_matrix=clean_distance
        )
        transformed = self.transf_mlp(moiety_embeddings)
        moiety_embeddings = self.transf_norm(transformed + moiety_embeddings)
        token_mask = x_flat.ne(0).to(moiety_embeddings.dtype).unsqueeze(-1)
        molecule_embeddings = (moiety_embeddings * token_mask).sum(dim=1).reshape(
            batch_size, molecule_count, self.d_model
        )
        molar_ratios = batch["molar_ratios"]
        molecule_types = batch["molecule_types"]
        if molar_ratios.shape != (batch_size, molecule_count):
            raise ValueError("Molar ratios do not match molecule slots")
        if molecule_types.shape != (batch_size, molecule_count):
            raise ValueError("Molecule types do not match molecule slots")
        if torch.any(molar_ratios < 0):
            raise ValueError("Molar ratios must be nonnegative")
        present = molar_ratios > 0
        if torch.any(~present.any(dim=1)):
            raise ValueError("Every formulation must contain a positive ratio")
        solvent_mask = present & molecule_types.eq(1)
        salt_mask = present & molecule_types.eq(2)
        if torch.any(~solvent_mask.any(dim=1)) or torch.any(~salt_mask.any(dim=1)):
            raise ValueError("Every formulation must contain solvent and salt")
        ratio_weights = self.molar_ratio_encoder(molar_ratios)
        scaled_embeddings = molecule_embeddings * ratio_weights
        solvent_out, solvent_attention = self.solvent_group(scaled_embeddings, solvent_mask)
        salt_out = self.salt_group(scaled_embeddings, salt_mask)
        combined = torch.stack([solvent_out, salt_out], dim=1)
        combined_mask = torch.stack([solvent_mask.any(dim=1), salt_mask.any(dim=1)], dim=1)
        attended, formulation_attention_weights = self.molecular_attention(combined, combined_mask)
        formulation = self.formulation_norm(self.global_pooling(attended, combined_mask))
        if not return_aux:
            return formulation
        return {
            "formulation": formulation,
            "molecule_embeddings": molecule_embeddings,
            "normalized_molar_ratios": ratio_weights[..., 0],
            "sanitized_adjacency": clean_adjacency,
            "sanitized_distance": clean_distance,
            "solvent_attention": solvent_attention,
            "formulation_attention_weights": formulation_attention_weights,
        }

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        formulation = self.encode_formulation(batch)
        temperature = batch["temperature_features"].to(formulation.dtype)
        if temperature.ndim != 2 or temperature.shape[1] != 2:
            raise ValueError("temperature_features must have shape [batch, 2]")
        if temperature.shape[0] != formulation.shape[0]:
            raise ValueError("Temperature batch size differs from formulation batch")
        intercept = self.arrhenius_intercept(formulation)
        temperature_slope = F.softplus(self.arrhenius_slope(formulation))
        base = intercept + temperature_slope * temperature[:, 1:2]
        residual = self.arrhenius_residual(torch.cat([formulation, temperature], dim=1))
        return {
            "log10_viscosity": base + residual,
            "base_log10_viscosity": base,
            "arrhenius_intercept": intercept,
            "temperature_slope": temperature_slope,
            "temperature_residual": residual,
        }


def build_model(vocab_size: int, head_mode: str = HEAD_MODE) -> HELENACETemperatureArrhenius:
    """Build the model with the published HELENA-CE architecture settings."""
    if head_mode not in CE_TEMPERATURE_HEAD_MODES:
        raise ValueError(f"Unsupported head mode: {head_mode!r}")
    return HELENACETemperatureArrhenius(
        input_vocab_size=vocab_size,
        max_num_molecules=8,
        head_mode=head_mode,
        **ARCHITECTURE,
    )


def load_ce_backbone(
    model: nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load every compatible HELENA-CE hierarchy tensor, never its CE head."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ce_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for key, value in ce_state.items():
        if key.startswith("ce_head."):
            skipped[key] = "CE-specific prediction head"
            continue
        if key not in model_state:
            skipped[key] = "missing from viscosity model"
            continue
        if model_state[key].shape == value.shape:
            transferred[key] = value
            continue
        if (
            key == "encoder_moiety.embedding.weight"
            and value.ndim == 2
            and value.shape[1] == model_state[key].shape[1]
        ):
            extended = model_state[key].clone()
            copy_rows = min(value.shape[0], extended.shape[0])
            extended[:copy_rows] = value[:copy_rows]
            transferred[key] = extended
            skipped[key] = f"extended embedding; copied {copy_rows} rows"
            continue
        skipped[key] = (
            f"shape mismatch {tuple(value.shape)} != {tuple(model_state[key].shape)}"
        )

    incompatible = model.load_state_dict(transferred, strict=False)
    missing_preserved = sorted(
        key
        for key in incompatible.missing_keys
        if key.startswith(PRESERVED_STATE_PREFIXES)
    )
    if missing_preserved or incompatible.unexpected_keys:
        raise RuntimeError(
            "Backbone transfer mismatch: "
            f"missing_preserved={missing_preserved}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    transferred_parameters = sum(model_state[key].numel() for key in transferred)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_vocab_size": checkpoint.get("vocab_size"),
        "transferred_state_tensors": len(transferred),
        "transferred_parameters": transferred_parameters,
        "total_parameters": total_parameters,
        "transferred_parameter_fraction": transferred_parameters / total_parameters,
        "skipped_ce_tensors": skipped,
        "preserved_hierarchy_complete": not missing_preserved,
    }
