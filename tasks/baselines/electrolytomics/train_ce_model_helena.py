"""
CE Model Training using LUMO Chemprop + PLSR Ensemble (Helena Dataset)

Complete training pipeline for helena_ce.csv format:
- Supports 5 solvents + 3 salts
- Uses concentration-based ratios
- Trains on log(1-CE) for fair comparison

This script:
1. Loads trained LUMO Chemprop model
2. Extracts 200-dim graph embeddings for all molecules
3. Combines embeddings with molecular descriptors
4. Creates stratified train/val/test splits (80%/10%/10%)
5. Trains PLSR ensemble with hyperparameter optimization
6. Saves trained models to ce_models_helena/ folder
"""

import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION: Change this seed to get different train/val/test splits
# ==============================================================================
RANDOM_SEED = 24  # Original seed used in paper. Try: 17, 42, 123, etc.
# ==============================================================================


class CEModelTrainerHelena:
    """Train PLSR ensemble for CE prediction using LUMO Chemprop embeddings (Helena format)"""
    
    def __init__(self, random_seed=RANDOM_SEED):
        script_dir = Path(__file__).parent.absolute()
        self.chemprop_path = script_dir / 'lumo_models' / 'fold_0' / 'model_0' / 'model.pt'
        self.ce_dataset_path = script_dir / 'ce_datasets' / 'helena_ce.csv'
        self.output_dir = script_dir / 'ce_datasets'
        self.model_dir = script_dir / 'ce_models_helena'
        self.results_dir = script_dir / 'results'
        self.chemprop_model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        
    def load_chemprop(self):
        """Load LUMO Chemprop model"""
        try:
            # Handle PyTorch 2.6 weights_only issue
            old_load = torch.load
            torch.load = lambda *a, **k: old_load(*a, **{**k, 'weights_only': False})
            
            from chemprop.utils import load_checkpoint
            self.chemprop_model = load_checkpoint(str(self.chemprop_path), device=self.device)
            
            torch.load = old_load
            print(f"✓ Loaded LUMO Chemprop model from {self.chemprop_path.relative_to(Path(__file__).resolve().parent)}")
            return True
        except Exception as e:
            print(f"✗ Error loading Chemprop: {e}")
            print("  Use the environment in tasks/baselines/electrolytomics/environment.yml")
            return False
    
    def get_embeddings(self, smiles_list):
        """Extract 200-dim embeddings from LUMO Chemprop model"""
        embeddings = []
        self.chemprop_model.eval()
        
        with torch.no_grad():
            for smiles in smiles_list:
                try:
                    batch = [[smiles]]
                    encoding = self.chemprop_model.encoder(batch)
                    embeddings.append(encoding.detach().cpu().numpy().flatten())
                except:
                    embeddings.append(np.zeros(200))
        
        return np.array(embeddings)
    
    def combine_embeddings_weighted(self, smiles_list, weights):
        """
        Get weighted average embedding for mixture
        
        Args:
            smiles_list: List of SMILES strings
            weights: List of weights (concentrations or ratios)
        """
        if not smiles_list or sum(weights) == 0:
            return np.zeros(200)
        
        # Get embeddings for all components
        embeddings = self.get_embeddings(smiles_list)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        combined = np.sum(embeddings * weights[:, np.newaxis], axis=0)
        return combined
    
    def extract_features(self, df):
        """
        Extract features from helena CE dataset
        
        Features:
        - 200-dim weighted average solvent embedding (from up to 5 solvents)
        - 200-dim weighted average salt embedding (from up to 3 salts)
        - N additional features (ratios, concentrations, descriptors, etc.)
        
        Total: 400 + N features
        """
        print("\nExtracting features from helena dataset...")
        features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            # Collect solvents and their concentrations
            solvent_smiles = []
            solvent_concs = []
            for i in range(1, 6):  # Up to 5 solvents
                smiles = row.get(f'Solvent_{i}', '')
                conc = row.get(f'Conc_Solvent_{i}', 0.0)
                if pd.notna(smiles) and smiles != '' and pd.notna(conc) and conc > 0:
                    solvent_smiles.append(smiles)
                    solvent_concs.append(float(conc))
            
            # Collect salts and their concentrations
            salt_smiles = []
            salt_concs = []
            for i in range(1, 4):  # Up to 3 salts
                smiles = row.get(f'Salt_{i}', '')
                conc = row.get(f'Conc_Salt_{i}', 0.0)
                if pd.notna(smiles) and smiles != '' and pd.notna(conc) and conc > 0:
                    salt_smiles.append(smiles)
                    salt_concs.append(float(conc))
            
            # Get weighted embeddings
            solvent_emb = self.combine_embeddings_weighted(solvent_smiles, solvent_concs)
            salt_emb = self.combine_embeddings_weighted(salt_smiles, salt_concs)
            
            # Calculate solvent ratios (normalized concentrations)
            total_solv_conc = sum(solvent_concs) if solvent_concs else 1.0
            solv_ratios = [c / total_solv_conc for c in solvent_concs]
            # Pad to 5 solvents
            while len(solv_ratios) < 5:
                solv_ratios.append(0.0)
            
            # Salt concentrations (pad to 3)
            salt_conc_padded = list(salt_concs)
            while len(salt_conc_padded) < 3:
                salt_conc_padded.append(0.0)
            
            # Additional features
            protocol = row.get('protocol', 0)
            current_density = row.get('current_density', 0.0)
            
            # Calculate molecular descriptors for first solvent (if available)
            if solvent_smiles:
                mol = Chem.MolFromSmiles(solvent_smiles[0])
                if mol:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    rotatable = Descriptors.NumRotatableBonds(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                else:
                    mw = logp = tpsa = rotatable = hbd = hba = 0.0
            else:
                mw = logp = tpsa = rotatable = hbd = hba = 0.0
            
            # Combine all features
            row_features = np.concatenate([
                solvent_emb,                          # 200 dims
                salt_emb,                              # 200 dims
                solv_ratios,                           # 5 dims
                salt_conc_padded,                      # 3 dims
                [protocol, current_density],           # 2 dims
                [mw, logp, tpsa, rotatable, hbd, hba]  # 6 dims
            ])
            
            features.append(row_features)
        
        features_array = np.array(features)
        print(f"✓ Extracted {features_array.shape[1]} features from {len(df)} samples")
        
        # Create dataframe with feature names
        feature_names = [f'feat_{i}' for i in range(features_array.shape[1])]
        df_features = pd.DataFrame(features_array, columns=feature_names)
        
        # Add original columns for splitting/tracking
        df_features['CE'] = df['CE'].values
        df_features['log(1-CE)'] = df['log(1-CE)'].values
        
        return df_features
    
    def create_ce_binned(self, ce_values, n_bins=5):
        """
        Create binned CE labels for stratified splitting
        Uses quantile-based binning to ensure roughly equal samples per bin
        """
        try:
            # Try quantile-based binning first (ensures equal distribution)
            ce_binned = pd.qcut(ce_values, q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            # Fall back to fewer bins if quantile binning fails
            print(f"  Warning: Could not create {n_bins} quantile bins, reducing to 4...")
            try:
                ce_binned = pd.qcut(ce_values, q=4, labels=False, duplicates='drop')
            except ValueError:
                # Last resort: simple binning
                print(f"  Warning: Using simple binning instead")
                ce_binned = pd.cut(ce_values, bins=4, labels=False)
        
        print(f"  Created {len(np.unique(ce_binned))} bins for stratification")
        print(f"  Samples per bin: {np.bincount(ce_binned.astype(int))}")
        return ce_binned
    
    def create_splits(self, df_features):
        """
        Create stratified train/val/test splits (80%/10%/10%)
        Following exact procedure from original notebook
        """
        seed = self.random_seed
        print(f"\nCreating stratified splits (seed={seed})...")
        
        # Create stratification labels
        ce_binned = self.create_ce_binned(df_features['CE'].values)
        df_features['CE_binned'] = ce_binned
        
        # Try stratified split, fall back to random if it fails
        try:
            # First split: separate test set (10%)
            df_rest, df_test, _, _ = train_test_split(
                df_features, 
                df_features['CE_binned'], 
                test_size=0.1, 
                random_state=seed, 
                stratify=df_features['CE_binned']
            )
            
            # Second split: separate val from train (10% of total)
            req_len = len(df_test)
            rat = req_len / len(df_rest)
            df_train, df_val, _, _ = train_test_split(
                df_rest, 
                df_rest['CE_binned'], 
                test_size=rat, 
                random_state=seed, 
                stratify=df_rest['CE_binned']
            )
            print(f"  ✓ Stratified split successful")
        except ValueError as e:
            print(f"  Warning: Stratified split failed ({e}), using random split")
            df_rest, df_test = train_test_split(df_features, test_size=0.1, random_state=seed)
            req_len = len(df_test)
            rat = req_len / len(df_rest)
            df_train, df_val = train_test_split(df_rest, test_size=rat, random_state=seed)
        
        print(f"  Train: {len(df_train)} samples ({len(df_train)/len(df_features)*100:.1f}%)")
        print(f"  Val:   {len(df_val)} samples ({len(df_val)/len(df_features)*100:.1f}%)")
        print(f"  Test:  {len(df_test)} samples ({len(df_test)/len(df_features)*100:.1f}%)")
        
        return df_train, df_val, df_test
    
    def prepare_data(self, df_train, df_val, df_test):
        """Prepare X and y, and scale last N features"""
        # Extract features and targets
        # Select only the feature columns (feat_0, feat_1, ..., feat_N)
        feature_cols = [col for col in df_train.columns if col.startswith('feat_')]
        X_train = df_train[feature_cols].values
        y_train = df_train['log(1-CE)'].values
        
        X_val = df_val[feature_cols].values
        y_val = df_val['log(1-CE)'].values
        
        X_test = df_test[feature_cols].values
        y_test = df_test['log(1-CE)'].values
        
        # Check for NaN values and replace with 0
        if np.isnan(X_train).any():
            n_nan_train = np.isnan(X_train).sum()
            print(f"  Warning: Found {n_nan_train} NaN values in train set, replacing with 0")
            X_train = np.nan_to_num(X_train, nan=0.0)
        
        if np.isnan(X_val).any():
            n_nan_val = np.isnan(X_val).sum()
            print(f"  Warning: Found {n_nan_val} NaN values in val set, replacing with 0")
            X_val = np.nan_to_num(X_val, nan=0.0)
        
        if np.isnan(X_test).any():
            n_nan_test = np.isnan(X_test).sum()
            print(f"  Warning: Found {n_nan_test} NaN values in test set, replacing with 0")
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        print(f"\n  Feature matrix shape: {X_train.shape}")
        print(f"  Training on: log(1-CE)")
        
        # Scale only the last features (not the embeddings)
        # Embeddings are first 400 dims, rest are hand-crafted features
        n_embed = 400
        n_features = X_train.shape[1]
        n_to_scale = n_features - n_embed
        
        print(f"  Scaling last {n_to_scale} features (embeddings unchanged)")
        
        # Fit scaler on train, transform all
        self.scaler.fit(X_train[:, n_embed:])
        X_train_scaled = np.copy(X_train)
        X_val_scaled = np.copy(X_val)
        X_test_scaled = np.copy(X_test)
        
        X_train_scaled[:, n_embed:] = self.scaler.transform(X_train[:, n_embed:])
        X_val_scaled[:, n_embed:] = self.scaler.transform(X_val[:, n_embed:])
        X_test_scaled[:, n_embed:] = self.scaler.transform(X_test[:, n_embed:])
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_plsr_ensemble(self, X_train, y_train, X_val, y_val, n_components=11, n_folds=5):
        """
        Train PLSR ensemble with cross-validation
        
        Following original paper:
        - n_components = 11 (optimized via grid search)
        - 5-fold ensemble
        """
        print(f"\nTraining PLSR ensemble (n_components={n_components}, n_folds={n_folds})...")
        
        models = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        # Create bins for stratification
        y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_binned), 1):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            
            plsr = PLSRegression(n_components=n_components, scale=False)
            plsr.fit(X_fold_train, y_fold_train)
            models.append(plsr)
            
            # Evaluate on fold validation
            y_fold_pred = plsr.predict(X_train[val_idx]).flatten()
            r2_fold = r2_score(y_train[val_idx], y_fold_pred)
            print(f"  Fold {fold}/5: R² = {r2_fold:.4f}")
        
        print(f"✓ Trained {len(models)} PLSR models")
        return models
    
    def predict_ensemble(self, models, X):
        """Make ensemble predictions (average of all models)"""
        predictions = np.array([model.predict(X).flatten() for model in models])
        return predictions.mean(axis=0)
    
    def evaluate(self, y_true, y_pred, split_name):
        """Calculate and print evaluation metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\n{split_name} Performance:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    
    def plot_parity(self, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, 
                    metrics_train, metrics_val, metrics_test, output_path):
        """
        Create parity plot with train/val/test colored separately
        
        Args:
            y_train, y_pred_train: Training data
            y_val, y_pred_val: Validation data
            y_test, y_pred_test: Test data
            metrics_*: Dictionary with 'rmse' key
            output_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot each split with different colors
        ax.scatter(y_train, y_pred_train, alpha=0.6, s=50, edgecolors='black', 
                  linewidths=0.5, label='Train', color='#F94144')  # Red/orange
        ax.scatter(y_val, y_pred_val, alpha=0.6, s=50, edgecolors='black', 
                  linewidths=0.5, label='Val', color='#8E44F9')     # Purple
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', 
                  linewidths=0.5, label='Test', color='#0093E9')    # Blue
        
        # Add diagonal line (x=y)
        all_vals = np.concatenate([y_train, y_val, y_test, y_pred_train, y_pred_val, y_pred_test])
        min_val = all_vals.min()
        max_val = all_vals.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)
        
        # Labels
        ax.set_xlabel('True LCE', fontsize=12)
        ax.set_ylabel('Predicted LCE', fontsize=12)
        
        # Legend in upper left
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        
        # Add RMSE values in lower right
        rmse_text = f"Train RMSE: {metrics_train['rmse']:.3f}\n"
        rmse_text += f"Val RMSE: {metrics_val['rmse']:.3f}\n"
        rmse_text += f"Test RMSE: {metrics_test['rmse']:.3f}"
        ax.text(0.95, 0.05, rmse_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
        
        # Add dataset label at top
        ax.text(0.5, 0.98, 'HELENA Dataset', transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='center',
                fontweight='bold')
        
        # Remove grid
        ax.grid(False)
        
        # Make axes equal
        ax.set_aspect('equal', adjustable='box')
        
        # Tight layout
        plt.tight_layout()
        
        # Save with high DPI
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved parity plot to {output_path.relative_to(Path(__file__).resolve().parent)}")
    
    def run(self):
        """Main training pipeline"""
        print("="*70)
        print("CE Model Training Pipeline (Helena Dataset)")
        print("="*70)
        
        # Step 1: Load Chemprop
        print(f"\nStep 1: Loading LUMO Chemprop Model")
        print("-" * 70)
        if not self.load_chemprop():
            return False
        
        # Step 2: Load CE dataset
        print(f"\nStep 2: Loading helena CE Dataset")
        print("-" * 70)
        df = pd.read_csv(self.ce_dataset_path)
        print(f"✓ Loaded {len(df)} CE samples from {self.ce_dataset_path.relative_to(Path(__file__).resolve().parent)}")
        print(f"  Columns: {list(df.columns[:10])}...")
        
        # Step 3: Calculate log(1-CE)
        print(f"\nStep 3: Calculating log(1-CE)")
        print("-" * 70)
        # Convert CE (%) to fraction, then calculate log base 10
        df['CE'] = df['CE (%)'] / 100.0
        df['log(1-CE)'] = np.log10(1 - df['CE'])
        print(f"✓ CE range: {df['CE'].min():.4f} - {df['CE'].max():.4f}")
        print(f"✓ log(1-CE) range: {df['log(1-CE)'].min():.4f} - {df['log(1-CE)'].max():.4f}")
        
        # Map protocol (Cycle=1, Aurbach=2 as per paper)
        protocol_mapping = {'Cycle': 1, 'Aurbach': 2}
        df['protocol'] = df['Method'].map(protocol_mapping).fillna(0).astype(int)
        print(f"✓ Protocol mapping: {protocol_mapping}")
        
        # Rename current column
        df['current_density'] = df['Current (mA/cm2)']
        
        # Step 4: Extract features
        print(f"\nStep 4: Extracting Features")
        print("-" * 70)
        df_features = self.extract_features(df)
        
        # Step 5: Create splits
        print(f"\nStep 5: Creating train/val/test splits")
        print("-" * 70)
        df_train, df_val, df_test = self.create_splits(df_features)
        
        # Save splits
        self.output_dir.mkdir(exist_ok=True, parents=True)
        df_train.to_csv(self.output_dir / 'helena_train_features.csv', index=False)
        df_val.to_csv(self.output_dir / 'helena_val_features.csv', index=False)
        df_test.to_csv(self.output_dir / 'helena_test_features.csv', index=False)
        print(f"✓ Saved splits to {self.output_dir.relative_to(Path(__file__).resolve().parent)}/")
        
        # Step 6: Prepare data
        print(f"\nStep 6: Preparing Data")
        print("-" * 70)
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df_train, df_val, df_test)
        
        # Step 7: Train PLSR ensemble
        print(f"\nStep 7: Training PLSR Ensemble")
        print("-" * 70)
        models = self.train_plsr_ensemble(X_train, y_train, X_val, y_val, n_components=11, n_folds=5)
        
        # Save models
        self.model_dir.mkdir(exist_ok=True, parents=True)
        for i, model in enumerate(models, 1):
            model_path = self.model_dir / f'plsr_ce_helena_ncomp_11_fold_{i}.sav'
            pickle.dump(model, open(model_path, 'wb'))
        print(f"✓ Saved {len(models)} models to {self.model_dir.relative_to(Path(__file__).resolve().parent)}/")
        
        # Save scaler
        scaler_path = self.model_dir / 'minmax_scaler.pkl'
        pickle.dump(self.scaler, open(scaler_path, 'wb'))
        print(f"✓ Saved scaler to {scaler_path.relative_to(Path(__file__).resolve().parent)}")
        
        # Step 8: Evaluate
        print(f"\nStep 8: Evaluation")
        print("-" * 70)
        y_pred_train = self.predict_ensemble(models, X_train)
        y_pred_val = self.predict_ensemble(models, X_val)
        y_pred_test = self.predict_ensemble(models, X_test)
        
        metrics_train = self.evaluate(y_train, y_pred_train, "Train")
        metrics_val = self.evaluate(y_val, y_pred_val, "Validation")
        metrics_test = self.evaluate(y_test, y_pred_test, "Test")
        
        # Create parity plot
        print(f"\nGenerating parity plot...")
        plot_path = self.results_dir / f'helena_training_parity_plot_{self.random_seed}.png'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.plot_parity(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test,
                        metrics_train, metrics_val, metrics_test, plot_path)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'split': ['train', 'val', 'test'],
            'r2': [metrics_train['r2'], metrics_val['r2'], metrics_test['r2']],
            'rmse': [metrics_train['rmse'], metrics_val['rmse'], metrics_test['rmse']],
            'mae': [metrics_train['mae'], metrics_val['mae'], metrics_test['mae']]
        })
        metrics_path = self.model_dir / 'training_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Saved metrics to {metrics_path.relative_to(Path(__file__).resolve().parent)}")
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"\nModels saved to: {self.model_dir.relative_to(Path(__file__).resolve().parent)}/")
        print(f"  - plsr_ce_helena_ncomp_11_fold_[1-5].sav")
        print(f"  - minmax_scaler.pkl")
        print(f"  - training_metrics.csv")
        print(f"\nData splits saved to: {self.output_dir.relative_to(Path(__file__).resolve().parent)}/")
        print(f"  - helena_train_features.csv")
        print(f"  - helena_val_features.csv")
        print(f"  - helena_test_features.csv")
        print(f"\nPlot saved to: {self.results_dir.relative_to(Path(__file__).resolve().parent)}/")
        print(f"  - helena_training_parity_plot_{self.random_seed}.png")
        print("\n" + "="*70 + "\n")
        
        return True


if __name__ == "__main__":
    trainer = CEModelTrainerHelena(random_seed=RANDOM_SEED)
    success = trainer.run()
    
    if not success:
        sys.exit(1)

