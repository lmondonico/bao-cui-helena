"""
Processing of the datasets into PyTorch tensors as model input data.
Batching and padding of the tensors to account for dimensionality mismatches.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.datasets import MAX_SOLVENTS, MAX_SALTS


class FeatureEngineering(Dataset):
    def __init__(
        self,
        data_set,
        tokenizer,
        map_dict,
        label_field,
        molar_ratio_noise,
    ):
        """
        PyTorch Dataset class for preparing data to feed into model.

        Args:
            data_set (pd.DataFrame): DataFrame containing the data
            tokenizer: Tokenizer instance
            map_dict (dict): Preprocessed mol information dictionary
            label_field (str): The label column name in the DataFrame (None for inference)
            molar_ratio_noise (float): Noise range for molar ratio values
        """
        self.data_set = data_set.copy()
        self.label_field = label_field

        # Only validate label column if label_field is provided (not None for inference)
        if label_field is not None:
            self.data_set[label_field] = self.data_set[label_field].astype(float)
        self.tokenizer = tokenizer
        self.pad_value = 0  # Reserve moiety ID 0 for padding
        self.map_dict = map_dict

        # CE dataset configuration
        self.max_num_solvent_molecules = MAX_SOLVENTS
        self.max_num_salt_molecules = MAX_SALTS
        self.max_num_molecules = (
            self.max_num_solvent_molecules + self.max_num_salt_molecules
        )
        self.molar_ratio_noise = molar_ratio_noise

        # Handle Method column - use provided values or default
        if "Method" in self.data_set.columns:
            self.data_set["Method"] = self.data_set["Method"].fillna("Aurbach")
            self.data_set["method_numeric"] = self.data_set["Method"].map(
                {"Aurbach": 0, "Cycle": 1}
            )
        else:
            self.data_set["method_numeric"] = 0

        # Handle Current column - use provided values or default
        if "Current" in self.data_set.columns:
            self.data_set["Current"] = self.data_set["Current"].fillna(0.5)
            self.data_set["current_numeric"] = self.data_set["Current"].astype(float)
        else:
            self.data_set["current_numeric"] = 0.5

        self.data_list = []

        self._process_data()

    def _add_molar_ratio_noise(self, molar_ratio):
        """
        Add noise to molar ratio value ensuring it stays within a specified percentage range.
        """
        # Skip noise generation entirely when noise is 0 (inference mode)
        if molar_ratio <= 0 or self.molar_ratio_noise == 0:
            return molar_ratio

        min_allowed = molar_ratio * (1 - self.molar_ratio_noise)
        max_allowed = molar_ratio * (1 + self.molar_ratio_noise)
        noisy_molar_ratio = np.random.uniform(min_allowed, max_allowed)

        return noisy_molar_ratio

    def _process_data(self):
        # Preprocess and store data tuples
        for idx, row in self.data_set.iterrows():
            smiles_list = []
            molar_ratios = []
            molecule_types = []  # 0=padding, 1=solvent, 2=salt

            # Process solvent molecules
            for i in range(1, self.max_num_solvent_molecules + 1):
                mol_col = f"Solvent_{i}"

                if mol_col in row and pd.notna(row[mol_col]):
                    # Parse SMILES:molar_ratio format
                    try:
                        parts = str(row[mol_col]).split(":")
                        if len(parts) == 2:
                            smiles = parts[0].strip()
                            molar_ratio = float(parts[1].strip())
                            smiles_list.append(smiles)
                            molar_ratios.append(molar_ratio)
                            molecule_types.append(1)  # Solvent type
                        else:
                            raise ValueError(
                                f"Invalid format in {mol_col}: expected 'SMILES:molar_ratio'"
                            )
                    except (ValueError, AttributeError) as e:
                        raise ValueError(f"Error parsing {mol_col} at index {idx}: {e}")

            # Process salt molecules
            for i in range(1, self.max_num_salt_molecules + 1):
                mol_col = f"Salt_{i}"

                if mol_col in row and pd.notna(row[mol_col]):
                    # Parse SMILES:molar_ratio format
                    try:
                        parts = str(row[mol_col]).split(":")
                        if len(parts) == 2:
                            smiles = parts[0].strip()
                            molar_ratio = float(parts[1].strip())
                            smiles_list.append(smiles)
                            molar_ratios.append(molar_ratio)
                            molecule_types.append(2)  # Salt type
                        else:
                            raise ValueError(
                                f"Invalid format in {mol_col}: expected 'SMILES:molar_ratio'"
                            )
                    except (ValueError, AttributeError) as e:
                        raise ValueError(f"Error parsing {mol_col} at index {idx}: {e}")

            # Ensure consistent lengths
            num_molecules = len(smiles_list)
            assert num_molecules == len(molar_ratios) == len(molecule_types), (
                f"Mismatch in lengths at index {idx}"
            )

            # Pad lists to max_num_molecules
            padding_length = self.max_num_molecules - num_molecules
            if padding_length > 0:
                smiles_list.extend([None] * padding_length)
                molar_ratios.extend([0.0] * padding_length)
                molecule_types.extend([0] * padding_length)  # Pad with type 0 (padding)

            # Use dummy label for inference mode (label_field=None)
            label = row[self.label_field] if self.label_field is not None else 0.0
            row_id = row["Row_ID"]  # Get Formulation ID
            method = row["method_numeric"]
            current = row["current_numeric"]

            self.data_list.append(
                (
                    smiles_list,
                    label,
                    molar_ratios,
                    molecule_types,
                    row_id,
                    method,
                    current,
                )
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        (
            smiles_list,
            label,
            molar_ratios,
            molecule_types,
            row_id,
            method,
            current,
        ) = self.data_list[idx]
        # Add noise to molar ratios during training
        noisy_molar_ratios = [self._add_molar_ratio_noise(mr) for mr in molar_ratios]
        sample = self.numerical_seq(
            smiles_list, label, noisy_molar_ratios, molecule_types
        )
        sample["row_id"] = row_id
        sample["method"] = torch.tensor(method, dtype=torch.long)
        sample["current"] = torch.tensor(current, dtype=torch.float32)
        return sample

    def numerical_seq(self, smiles_list, label, molar_ratios, molecule_types):
        """
        Process molecules to generate numerical sequences and features.

        Args:
            smiles_list: List of SMILES strings (possibly with None for padding)
            label: Target value
            molar_ratios: List of molar ratio values
            molecule_types: List of molecule types (0=padding, 1=solvent, 2=salt)
        """
        sample = {}
        num_mols = sum(1 for smiles in smiles_list if smiles is not None)
        sample["num_mols"] = num_mols

        for i in range(1, self.max_num_molecules + 1):
            if i <= num_mols and smiles_list[i - 1] is not None:
                smiles = smiles_list[i - 1]
                # Process molecule i
                moiety_ids = self.map_dict[smiles]["moiety_ids"]
                dist_matrix = self.map_dict[smiles]["dist_matrix"]
                adjoin_matrix = self.map_dict[smiles]["adj_matrix"]

                sample[f"x{i}"] = np.array(moiety_ids, dtype=np.int64)
                sample[f"adjoin_matrix{i}"] = adjoin_matrix
                sample[f"dist_matrix{i}"] = dist_matrix
                sample[f"mr{i}"] = float(molar_ratios[i - 1])
                sample[f"mol_type{i}"] = int(molecule_types[i - 1])
            else:
                # Pad with default values
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
        """
        Collate function to be used with DataLoader for padding and batching.
        """
        max_num_mols = self.max_num_mols
        collated_batch = {"num_mols": max_num_mols}

        # Initialize lists to collect data for each mol
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
                    collated_batch[f"x{i}"].append(
                        torch.as_tensor(sample[f"x{i}"], dtype=torch.int64)
                    )
                    collated_batch[f"adjoin_matrix{i}"].append(
                        torch.as_tensor(
                            sample[f"adjoin_matrix{i}"], dtype=torch.float32
                        )
                    )
                    collated_batch[f"dist_matrix{i}"].append(
                        torch.as_tensor(sample[f"dist_matrix{i}"], dtype=torch.float32)
                    )
                    collated_batch[f"mr{i}"].append(
                        torch.as_tensor(
                            sample[f"mr{i}"], dtype=torch.float32
                        ).unsqueeze(0)
                    )
                    collated_batch[f"mol_type{i}"].append(
                        torch.as_tensor(
                            sample[f"mol_type{i}"], dtype=torch.long
                        ).unsqueeze(0)
                    )
                else:
                    # Pad with default values
                    collated_batch[f"x{i}"].append(
                        torch.as_tensor([0], dtype=torch.int64)
                    )
                    collated_batch[f"adjoin_matrix{i}"].append(
                        torch.as_tensor([[-1e9]], dtype=torch.float32)
                    )
                    collated_batch[f"dist_matrix{i}"].append(
                        torch.as_tensor([[-1e9]], dtype=torch.float32)
                    )
                    collated_batch[f"mr{i}"].append(
                        torch.as_tensor([0.0], dtype=torch.float32)
                    )
                    collated_batch[f"mol_type{i}"].append(
                        torch.as_tensor([0], dtype=torch.long)
                    )

        # Stack all tensors in the batch
        for i in range(1, max_num_mols + 1):
            collated_batch[f"x{i}"] = self.pad_sequence_list(
                collated_batch[f"x{i}"], padding_value=0
            )
            collated_batch[f"adjoin_matrix{i}"] = self.pad_matrix_list(
                collated_batch[f"adjoin_matrix{i}"], padding_value=-1e9
            )
            collated_batch[f"dist_matrix{i}"] = self.pad_matrix_list(
                collated_batch[f"dist_matrix{i}"], padding_value=-1e9
            )
            collated_batch[f"mr{i}"] = torch.cat(collated_batch[f"mr{i}"], dim=0)
            collated_batch[f"mol_type{i}"] = torch.cat(
                collated_batch[f"mol_type{i}"], dim=0
            )

        collated_batch["y"] = torch.cat(y_list, dim=0)
        collated_batch["row_id"] = torch.tensor(row_id_list)
        collated_batch["method"] = torch.stack(method_list)
        collated_batch["current"] = torch.stack(current_list)
        return collated_batch

    def pad_sequence_list(self, sequence_list, padding_value=0):
        if not sequence_list:
            return torch.empty(0)
        return torch.nn.utils.rnn.pad_sequence(
            sequence_list, batch_first=True, padding_value=padding_value
        )

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
            m_padded = torch.nn.functional.pad(
                m_tensor, (0, pad_width, 0, pad_height), value=padding_value
            )
            padded_matrices.append(m_padded)

        return torch.stack(padded_matrices)
