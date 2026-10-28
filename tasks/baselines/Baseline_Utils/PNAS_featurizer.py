"""
Functions to obtain the atomic ratios from PNAS paper
"""

import numpy as np
import pandas as pd
from rdkit import Chem


class PNASFeaturizer:
    def __init__(self):
        self.inorganic_elements = set(["F", "O", "N", "S", "P", "B", "Si", "Cl", "Br", "I"])

    def count_atoms(self, smiles):
        """Count all atoms including implicit hydrogens in a molecule"""
        mol = Chem.MolFromSmiles(smiles)
        total_atoms = 0
        for atom in mol.GetAtoms():
            total_atoms += 1 + atom.GetTotalNumHs()  # Explicit + implicit H
        return total_atoms

    def count_element(self, smiles, element):
        """Count occurrences of a specific element in a molecule"""
        mol = Chem.MolFromSmiles(smiles)
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)

    def count_organic_inorganic(self, smiles):
        """Count organic (C,H) vs inorganic atoms"""
        mol = Chem.MolFromSmiles(smiles)
        organic = 0
        inorganic = 0

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            num_h = atom.GetTotalNumHs()

            if symbol == "C":
                organic += 1 + num_h
            elif symbol in self.inorganic_elements:
                inorganic += 1
            # Implicit H already counted with C

        return organic, inorganic

    def _parse_smiles_conc(self, value):
        """Parse 'SMILES:concentration' format used in HELENA dataset.

        Returns (smiles, concentration) tuple or (None, 0) if invalid.
        """
        if pd.isna(value):
            return None, 0.0

        value = str(value)
        if ":" in value:
            parts = value.split(":")
            if len(parts) == 2:
                smiles = parts[0].strip()
                try:
                    conc = float(parts[1].strip())
                    return smiles, conc
                except ValueError:
                    return smiles, 1.0
        # Fallback: treat as SMILES only
        return value, 1.0

    def calculate_features(self, sample):
        """
        Calculate all four PNAS features for a sample.

        Handles the HELENA dataset format where molecules are stored as 'SMILES:molar_ratio'
        (e.g., 'C1COC(=O)O1:4.49683177761123').
        """
        features = {}

        # Initialize counters
        solvent_O = 0
        solvent_total_atoms = 0
        total_F = 0
        total_O = 0
        total_organic = 0
        total_inorganic = 0
        salt_C = 0
        salt_total = 0

        # Process solvents (columns contain 'SMILES:molar_ratio' format)
        for i in range(1, 6):
            mol_key = f"Solvent_{i}"

            if mol_key in sample and pd.notna(sample[mol_key]):
                smiles, conc = self._parse_smiles_conc(sample[mol_key])
                if smiles is None:
                    continue

                # For solvent oxygen ratio
                solvent_O += self.count_element(smiles, "O") * conc
                solvent_total_atoms += self.count_atoms(smiles) * conc

                # For overall F/O ratio and In/Or ratio
                total_F += self.count_element(smiles, "F") * conc
                total_O += self.count_element(smiles, "O") * conc
                org, inorg = self.count_organic_inorganic(smiles)
                total_organic += org * conc
                total_inorganic += inorg * conc

        # Process salts (columns contain 'SMILES:molar_ratio' format)
        for i in range(1, 4):
            mol_key = f"Salt_{i}"

            if mol_key in sample and pd.notna(sample[mol_key]):
                smiles, conc = self._parse_smiles_conc(sample[mol_key])
                if smiles is None:
                    continue

                # For F/O ratio and In/Or ratio
                total_F += self.count_element(smiles, "F") * conc
                total_O += self.count_element(smiles, "O") * conc
                org, inorg = self.count_organic_inorganic(smiles)
                total_organic += org * conc
                total_inorganic += inorg * conc

                # For anion carbon ratio
                salt_C += self.count_element(smiles, "C") * conc
                salt_total += self.count_atoms(smiles) * conc  # excluding Li+

        # Calculate final ratios
        features["sO"] = solvent_O / solvent_total_atoms if solvent_total_atoms > 0 else 0
        features["FO"] = total_F / total_O if total_O > 0 else 0
        features["InOr"] = total_inorganic / total_organic if total_organic > 0 else 0
        features["aC"] = salt_C / salt_total if salt_total > 0 else 0

        # Add current density (log-transformed) and method
        if "Current" in sample:
            features["log_current"] = np.log10(sample["Current"] + 1e-8)
        else:
            features["log_current"] = 0.0

        if "Method" in sample:
            method_val = sample["Method"]
            if isinstance(method_val, str):
                features["method"] = 1.0 if method_val == "Cycle" else 0.0
            else:
                features["method"] = float(method_val)
        else:
            features["method"] = 0.0

        return features
