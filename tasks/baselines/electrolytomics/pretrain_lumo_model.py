"""
Step 1: Train Chemprop model on LUMO dataset

This creates the pretrained Chemprop model that will be used to generate
graph embeddings for CE prediction.

Based on paper specs:
- Neural network size: 200
- Depth of message passing layer: 3  
- Dropout: 0.05
- Number of hidden layers: 3
- Target: LUMO values
- Expected test R²: 0.98
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys
import warnings
from tqdm import tqdm

# Suppress pkg_resources warnings
warnings.filterwarnings('ignore', message='.*pkg_resources.*')

def prepare_lumo_data():
    """Load LUMO_clean.csv dataset"""
    print("="*60)
    print("Preparing LUMO Dataset")
    print("="*60 + "\n")
    
    # Use LUMO_clean.csv from the lumo_datasets/ subfolder
    script_dir = Path(__file__).parent.absolute()
    lumo_file = script_dir / 'lumo_datasets' / 'LUMO_clean.csv'
    
    print(f"Looking for: {lumo_file}")
    
    if not lumo_file.exists():
        print(f"ERROR: {lumo_file} not found!")
        print("Expected location: lumo_datasets/LUMO_clean.csv")
        return False
    
    print(f"Loading {lumo_file}...")
    chemprop_df = pd.read_csv(lumo_file)
    print(f"✓ Loaded {len(chemprop_df)} samples\n")
    
    # Check columns
    if 'SMILES' not in chemprop_df.columns or 'LUMO' not in chemprop_df.columns:
        print(f"ERROR: Expected columns 'SMILES' and 'LUMO'")
        print(f"Found columns: {list(chemprop_df.columns)}")
        return False
    
    # Rename to chemprop format (lowercase)
    chemprop_df = chemprop_df.rename(columns={'SMILES': 'smiles', 'LUMO': 'lumo'})
    chemprop_df = chemprop_df[['smiles', 'lumo']]
    
    print(f"Columns: {list(chemprop_df.columns)}")
    
    # Remove any NaN values
    initial_len = len(chemprop_df)
    chemprop_df = chemprop_df.dropna()
    if len(chemprop_df) < initial_len:
        print(f"Removed {initial_len - len(chemprop_df)} rows with missing values")
    
    print(f"Final dataset: {len(chemprop_df)} samples")
    print(f"LUMO range: [{chemprop_df['lumo'].min():.3f}, {chemprop_df['lumo'].max():.3f}]")
    print(f"LUMO mean: {chemprop_df['lumo'].mean():.3f} ± {chemprop_df['lumo'].std():.3f}\n")
    
    # Create train/val/test splits (80/10/10)
    from sklearn.model_selection import train_test_split
    
    print("Creating train/val/test splits (80/10/10)...")
    train_val, test = train_test_split(chemprop_df, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 of 90% = 10% total
    
    print("Data splits:")
    print(f"  Train: {len(train)} samples ({len(train)/len(chemprop_df)*100:.1f}%)")
    print(f"  Val:   {len(val)} samples ({len(val)/len(chemprop_df)*100:.1f}%)")
    print(f"  Test:  {len(test)} samples ({len(test)/len(chemprop_df)*100:.1f}%)\n")
    
    # Save splits to the lumo_datasets/ folder
    output_dir = script_dir / 'lumo_datasets'
    train.to_csv(output_dir / 'lumo_train.csv', index=False)
    val.to_csv(output_dir / 'lumo_val.csv', index=False)
    test.to_csv(output_dir / 'lumo_test.csv', index=False)
    
    print(f"✓ Saved train/val/test splits to {output_dir}/\n")
    return True


def train_chemprop():
    """Train Chemprop model on LUMO data"""
    print("="*60)
    print("Training Chemprop Model")
    print("="*60 + "\n")
    
    # Check if chemprop is installed
    try:
        import chemprop
        print(f"✓ Chemprop installed (version {chemprop.__version__})\n")
    except ImportError:
        print("ERROR: Chemprop not installed")
        print("Use the environment in tasks/baselines/electrolytomics/environment.yml\n")
        return False
    
    print("Training with paper hyperparameters:")
    print("  - Hidden size: 200")
    print("  - Depth: 3")
    print("  - Dropout: 0.05")
    print("  - FFN layers: 3")
    print("  - Target: R² = 0.98\n")
    
    # Use subprocess with Python -c to apply sklearn patch
    train_script = """
import sys
import warnings
warnings.filterwarnings('ignore')

# Patch torch.load for PyTorch 2.6 compatibility
import torch
_orig_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Patch sklearn for chemprop compatibility
from sklearn import metrics
_orig_mse = metrics.mean_squared_error

def patched_mse(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True):
    try:
        return _orig_mse(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=squared)
    except TypeError:
        result = _orig_mse(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
        return result if squared else result ** 0.5

metrics.mean_squared_error = patched_mse

if __name__ == '__main__':
    # Now run chemprop
    from chemprop.train import chemprop_train
    sys.argv = ['chemprop_train',
        '--data_path', 'lumo_datasets/lumo_train.csv',
        '--separate_val_path', 'lumo_datasets/lumo_val.csv',
        '--separate_test_path', 'lumo_datasets/lumo_test.csv',
        '--dataset_type', 'regression',
        '--save_dir', 'lumo_models',
        '--hidden_size', '200',
        '--depth', '3',
        '--dropout', '0.05',
        '--ffn_num_layers', '3',
        '--epochs', '50',
        '--batch_size', '50',
        '--num_workers', '0',
        '--show_individual_scores',
        '--save_preds',
        '--metric', 'mae',
        '--extra_metrics', 'r2']
    chemprop_train()
"""
    
    print("Starting training with compatibility patch...")
    print("(This will take ~1-2 hours for 77k samples)\n")
    
    result = subprocess.run(
        [sys.executable, '-c', train_script], cwd=Path(__file__).resolve().parent
    )
    
    if result.returncode == 0:
        print("\n✓ Training completed!")
        return True
    else:
        print("\n✗ Training failed")
        return False


def main():
    print("\n" + "="*60)
    print("Step 1: Train LUMO Chemprop Model")
    print("="*60 + "\n")
    
    print("This script will:")
    print("1. Load LUMO_clean.csv from lumo_datasets/")
    print("2. Prepare train/val/test splits")
    print("3. Train Chemprop model with paper specifications")
    print("4. Save model to lumo_models/\n")
    
    input("Press Enter to start (or Ctrl+C to cancel)...")
    print()
    
    # Step 1: Prepare data
    if not prepare_lumo_data():
        print("\n✗ Data preparation failed")
        sys.exit(1)
    
    # Step 2: Train model
    if not train_chemprop():
        print("\n✗ Model training failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ Step 1 Complete!")
    print("="*60)
    print("\nModel location: lumo_models/")
    print("  - Use lumo_models/fold_0/model_0/model.pt for embeddings extraction")
    print("\nNext steps:")
    print("  2. Run evaluate_lumo_model.py to evaluate the LUMO model")
    print("  3. Run train_ce_model.py to extract embeddings and train the CE models")
    print()


if __name__ == "__main__":
    main()

