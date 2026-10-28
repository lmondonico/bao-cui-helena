"""
Extract SMILES from SDF files and combine with LUMO data

This script:
1. Reads LUMO.xlsx for Filename, HOMO, LUMO values
2. Extracts SMILES from corresponding SDF files in Dataset/
3. Creates LUMO.csv with: Filename, SMILES, HOMO, LUMO
"""

import pandas as pd
from pathlib import Path
from rdkit import Chem
from tqdm import tqdm
import sys

def extract_smiles_from_sdf(sdf_path):
    """Extract SMILES from an SDF file"""
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path))
        for mol in supplier:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                return smiles
        return None
    except Exception as e:
        return None

def main():
    print("="*60)
    print("LUMO Dataset: Extract SMILES from SDF files")
    print("="*60 + "\n")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    print(f"Script location: {script_dir}\n")
    
    # Paths relative to script location
    xlsx_path = script_dir / 'LUMO.xlsx'
    dataset_dir = script_dir / 'sdf'  # Changed from 'Dataset' to 'sdf'
    output_path = script_dir / 'LUMO.csv'
    
    # Check files exist
    if not xlsx_path.exists():
        print(f"ERROR: {xlsx_path} not found!")
        sys.exit(1)
    
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir}/ not found!")
        print("Expected SDF files in: lumo_datasets/sdf/")
        sys.exit(1)
    
    # Load LUMO data
    print(f"Loading {xlsx_path}...")
    df = pd.read_excel(xlsx_path)
    print(f"✓ Loaded {len(df)} entries")
    print(f"  Columns: {list(df.columns)}\n")
    
    # Check for SDF files
    sdf_files = list(dataset_dir.glob('*.sdf'))
    print(f"Found {len(sdf_files)} SDF files in {dataset_dir}/\n")
    
    if len(sdf_files) == 0:
        print("ERROR: No SDF files found!")
        sys.exit(1)
    
    # Create filename to path mapping
    sdf_dict = {}
    for sdf_path in sdf_files:
        # Remove .sdf extension to match filename
        filename = sdf_path.stem
        sdf_dict[filename] = sdf_path
    
    print(f"Extracting SMILES from SDF files...")
    print("This may take a while...\n")
    
    # Extract SMILES for each entry
    smiles_list = []
    missing_files = []
    failed_extractions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        filename = row['Filename']
        
        if filename in sdf_dict:
            smiles = extract_smiles_from_sdf(sdf_dict[filename])
            if smiles:
                smiles_list.append(smiles)
            else:
                smiles_list.append(None)
                failed_extractions.append(filename)
        else:
            smiles_list.append(None)
            missing_files.append(filename)
    
    # Add SMILES column
    df['SMILES'] = smiles_list
    
    # Reorder columns: Filename, SMILES, HOMO, LUMO
    df = df[['Filename', 'SMILES', 'HOMO(eV)', 'LUMO(eV)']]
    df.columns = ['Filename', 'SMILES', 'HOMO', 'LUMO']
    
    # Statistics
    print("\n" + "="*60)
    print("Extraction Results")
    print("="*60)
    print(f"Total entries: {len(df)}")
    print(f"Successful: {df['SMILES'].notna().sum()}")
    print(f"Missing SDF files: {len(missing_files)}")
    print(f"Failed extractions: {len(failed_extractions)}")
    
    if missing_files:
        print(f"\nFirst 10 missing files:")
        for f in missing_files[:10]:
            print(f"  - {f}")
    
    if failed_extractions:
        print(f"\nFirst 10 failed extractions:")
        for f in failed_extractions[:10]:
            print(f"  - {f}")
    
    # Remove rows without SMILES
    df_clean = df.dropna(subset=['SMILES'])
    print(f"\nFinal dataset (after removing failures): {len(df_clean)} entries")
    
    # Save to CSV
    output_path_clean = script_dir / 'LUMO_clean.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Full dataset saved to: {output_path}")
    
    if not df_clean.equals(df):
        df_clean.to_csv(output_path_clean, index=False)
        print(f"✓ Clean dataset (no missing SMILES) saved to: {output_path_clean}")
    
    # Show sample
    print("\n" + "="*60)
    print("Sample Data")
    print("="*60)
    print(df_clean.head(10))
    
    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)
    print(f"\nYou can now use {output_path} for training Chemprop")


if __name__ == "__main__":
    main()

