"""
Module for tokenizing molecules into individual atoms.
"""

import os
import sys

import colorsys
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tasks.postprocess.tokenizer_studies.tokenizer_atoms_dict import ATOMS


class TokenizerAtoms:
    def __init__(self):
        self.atom_dict = ATOMS
        # Assign unique numeric keys to each atom type
        # We start the key from number 1 as we have 0 for padding
        self.moiety_key_map = {
            name: idx + 1 for idx, name in enumerate(self.atom_dict.keys())
        }
        self.key_moiety_map = {idx: name for name, idx in self.moiety_key_map.items()}
        # Build reverse lookup from atomic number to key
        self._atomic_num_to_key = self._build_atomic_num_map()
        # Initialize global color map
        self.global_color_map = {}

    def _build_atomic_num_map(self):
        """Build a mapping from atomic number to moiety key."""
        atomic_num_to_key = {}
        for name, smarts in self.atom_dict.items():
            # Extract atomic number from SMARTS like "[#6]"
            if smarts.startswith("[#") and smarts.endswith("]"):
                try:
                    atomic_num = int(smarts[2:-1])
                    atomic_num_to_key[atomic_num] = self.moiety_key_map[name]
                except ValueError:
                    pass
        return atomic_num_to_key

    def generate_single_color(self, key):
        """
        Generate a unique color based on the atom key using a combination of hue, saturation, and value.
        """
        hue = (key * 0.618033988749895) % 1
        saturation = 0.6 + 0.4 * (key % 2)
        value = 0.95
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def highlight_and_display_molecule(self, mol, matches, save_path=None):
        """
        Assign colors to each atom type and display the molecule.
        """
        highlight_atom_colors = {}
        for atom_name, atom_indices in matches:
            key = self.moiety_key_map.get(atom_name, -1)
            color = (
                (0.0, 0.0, 0.0)
                if key == -1
                else self.global_color_map.get(key, self.generate_single_color(key))
            )
            for atom in atom_indices:
                highlight_atom_colors[atom] = color

        drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
        options = drawer.drawOptions()
        options.addAtomIndices = True
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atom_colors.keys()),
            highlightAtomColors=highlight_atom_colors,
        )
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()

        image = Image.open(BytesIO(img))

        if save_path:
            image.save(save_path, dpi=(900, 900))
        else:
            image.show()

    def get_moieties(self, smiles):
        """
        Get atoms from a SMILES string.
        Each atom becomes its own "moiety" for compatibility with downstream code.

        Returns same format as Tokenizer.get_moieties():
            clique_atom_ids: list of lists, each containing a single atom index
            edges: list of tuples representing bonds between atoms
            moiety_ids: list of numeric keys for each atom type
            matches: list of (atom_name, {atom_idx}) tuples
            is_in_ring: list of 0/1 indicating if atom is in a ring
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES string.")
            return

        # Get ring atoms
        ring_info = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)

        # Process each atom
        clique_atom_ids = []
        moiety_ids = []
        matches = []
        is_in_ring = []

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()

            # Get the key for this atom type
            key = self._atomic_num_to_key.get(atomic_num, -1)

            # Get the atom name
            if key != -1:
                atom_name = self.key_moiety_map[key]
            else:
                # Unknown atom type - use element symbol
                atom_name = atom.GetSymbol().lower()
                # Add to maps if not present
                if atom_name not in self.moiety_key_map:
                    new_key = max(self.moiety_key_map.values()) + 1 if self.moiety_key_map else 1
                    self.moiety_key_map[atom_name] = new_key
                    self.key_moiety_map[new_key] = atom_name
                    self._atomic_num_to_key[atomic_num] = new_key
                    key = new_key
                else:
                    key = self.moiety_key_map[atom_name]

            clique_atom_ids.append([atom_idx])
            moiety_ids.append(key)
            matches.append((atom_name, {atom_idx}))
            is_in_ring.append(1 if atom_idx in ring_atoms else 0)

        # Generate edges from bonds
        edges = []
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edges.append(tuple(sorted((begin, end))))
        edges = sorted(edges)

        return clique_atom_ids, edges, moiety_ids, matches, is_in_ring

    def get_vocab_size(self):
        """Get the vocabulary size for this tokenizer."""
        if self.moiety_key_map:
            return max(self.moiety_key_map.values()) + 1
        else:
            return 1

    def encode(self, smiles):
        """Encode a SMILES string into token IDs."""
        try:
            clique_atom_ids, edges, moiety_ids, matches, is_in_ring = self.get_moieties(smiles)
            return moiety_ids
        except:
            return [0]  # Return padding token for invalid SMILES
