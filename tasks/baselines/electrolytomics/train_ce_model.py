"""
CE Model Training using LUMO Chemprop + PLSR Ensemble

Complete training pipeline: Raw SMILES → LUMO Embeddings → Features → PLSR Training

This script:
1. Loads trained LUMO Chemprop model
2. Extracts 200-dim graph embeddings for CE molecules
3. Combines with molecular descriptors (413 features total)
4. Creates stratified train/val/test splits (80%/10%/10%)
5. Trains PLSR ensemble with hyperparameter optimization
6. Saves trained models to ce_models/ folder
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
RANDOM_SEED = 11  # Original seed used in paper. Try: 17, 42, 123, etc.
# ==============================================================================


class CEModelTrainer:
    """Train PLSR ensemble for CE prediction using LUMO Chemprop embeddings"""
    
    def __init__(self, random_seed=RANDOM_SEED):
        script_dir = Path(__file__).parent.absolute()
        self.chemprop_path = script_dir / 'lumo_models' / 'fold_0' / 'model_0' / 'model.pt'
        self.ce_dataset_path = script_dir / 'ce_datasets' / 'EDB-2_ce.csv'
        self.output_dir = script_dir / 'ce_datasets'
        self.model_dir = script_dir / 'ce_models'
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
    
    def combine_embeddings(self, smiles_list):
        """Get combined embedding for mixture (uses first component)"""
        if not smiles_list:
            return np.zeros(200)
        emb = self.get_embeddings([smiles_list[0]])
        return emb[0]
    
    def extract_features(self, df):
        """
        Extract 413 features from CE dataset
        
        Features:
        - 200: solvent embeddings (LUMO Chemprop)
        - 200: salt embeddings (LUMO Chemprop)
        - 3: solvent ratios
        - 6: solvent descriptors (heavy atoms, MW)
        - 2: salt concentrations
        - 4: salt descriptors
        - 2: additive descriptors
        - 2: experimental conditions (protocol, current density)
        Total: 413 features
        """
        features_list = []
        
        print("Extracting graph embeddings from LUMO Chemprop model...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            features = []
            
            # 1. Solvent embeddings (200-dim)
            solvs = [row.get('solvent_1_smiles', ''), 
                    row.get('solvent_2_smiles', ''),
                    row.get('solvent_3_smiles', '')]
            solvs = [s for s in solvs if pd.notna(s) and s != '']
            solv_emb = self.combine_embeddings(solvs)
            features.extend(solv_emb)
            
            # 2. Salt embeddings (200-dim)
            salts = [row.get('salt_1_smiles', ''), row.get('salt_2_smiles', '')]
            salts = [s for s in salts if pd.notna(s) and s != '']
            salt_emb = self.combine_embeddings(salts)
            features.extend(salt_emb)
            
            # 3. Solvent ratios (3)
            features.extend([
                row.get('solv_1_ratio', 0.0),
                row.get('solv_2_ratio', 0.0),
                row.get('solv_3_ratio', 0.0)
            ])
            
            # 4. Solvent descriptors (6)
            for i in [1, 2, 3]:
                smiles = row.get(f'solvent_{i}_smiles', '')
                if pd.notna(smiles) and smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        features.append(mol.GetNumHeavyAtoms())
                        features.append(Descriptors.MolWt(mol))
                    else:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0, 0.0])
            
            # 5. Salt concentrations (2)
            features.extend([
                row.get('salt_1_conc', 0.0),
                row.get('salt_2_conc', 0.0)
            ])
            
            # 6. Salt descriptors (4)
            for i in [1, 2]:
                smiles = row.get(f'salt_{i}_smiles', '')
                if pd.notna(smiles) and smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        features.append(mol.GetNumHeavyAtoms())
                        features.append(Descriptors.MolWt(mol))
                    else:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0, 0.0])
            
            # 7. Additive descriptors (2)
            add_smiles = row.get('additive_smiles', '')
            if pd.notna(add_smiles) and add_smiles:
                mol = Chem.MolFromSmiles(add_smiles)
                if mol:
                    features.append(mol.GetNumHeavyAtoms())
                    features.append(Descriptors.MolWt(mol))
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # 8. Experimental conditions (2)
            features.append(row.get('protocol', 0))
            features.append(row.get('current_density', 0.0))
            
            features_list.append(features)
        
        features_array = np.array(features_list)
        print(f"✓ Extracted {features_array.shape[1]} features from {features_array.shape[0]} samples")
        
        return features_array
    
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
                # Last resort: very coarse binning
                print(f"  Warning: Using coarse binning with 3 bins...")
                ce_binned = pd.qcut(ce_values, q=3, labels=False, duplicates='drop')
        
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
        except ValueError as e:
            print(f"\n  Warning: Stratified split failed ({e})")
            print(f"  Falling back to random split (no stratification)...")
            
            # Fall back to random split
            df_rest, df_test = train_test_split(
                df_features,
                test_size=0.1,
                random_state=seed
            )
            
            req_len = len(df_test)
            rat = req_len / len(df_rest)
            df_train, df_val = train_test_split(
                df_rest,
                test_size=rat,
                random_state=seed
            )
        
        print(f"  Train: {len(df_train)} samples ({len(df_train)/len(df_features)*100:.1f}%)")
        print(f"  Val:   {len(df_val)} samples ({len(df_val)/len(df_features)*100:.1f}%)")
        print(f"  Test:  {len(df_test)} samples ({len(df_test)/len(df_features)*100:.1f}%)")
        
        # Save splits
        df_train.to_csv(self.output_dir / 'train_features.csv', index=False)
        df_val.to_csv(self.output_dir / 'val_features.csv', index=False)
        df_test.to_csv(self.output_dir / 'test_features.csv', index=False)
        print(f"✓ Saved splits to {self.output_dir.relative_to(Path(__file__).resolve().parent)}/")
        
        return df_train, df_val, df_test
    
    def prepare_data(self, df_train, df_val, df_test):
        """Prepare X and y, and scale last 13 features"""
        # Extract features and targets
        # Select only the feature columns (feat_0, feat_1, ..., feat_N)
        feature_cols = [col for col in df_train.columns if col.startswith('feat_')]
        
        X_train = df_train[feature_cols].values
        y_train = df_train['log(1-CE)'].values
        
        X_val = df_val[feature_cols].values
        y_val = df_val['log(1-CE)'].values
        
        X_test = df_test[feature_cols].values
        y_test = df_test['log(1-CE)'].values
        
        # Scale last 13 features (non-embedding features)
        print("\nScaling last 13 features with MinMaxScaler...")
        self.scaler.fit(X_train[:, -13:])
        
        X_train_scaled = self.scaler.transform(X_train[:, -13:])
        X_train = np.concatenate([X_train[:, :-13], X_train_scaled], axis=1)
        
        X_val_scaled = self.scaler.transform(X_val[:, -13:])
        X_val = np.concatenate([X_val[:, :-13], X_val_scaled], axis=1)
        
        X_test_scaled = self.scaler.transform(X_test[:, -13:])
        X_test = np.concatenate([X_test[:, :-13], X_test_scaled], axis=1)
        
        print(f"✓ Data prepared: X_train shape = {X_train.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def optimize_cv_splits(self, X_train, y_train, ce_binned, n_components=11):
        """
        Optimize StratifiedKFold random state by trying 100 options
        Select the one with lowest CV RMSE
        """
        print(f"\nOptimizing CV splits (trying 100 random states)...")
        print(f"Using {n_components} PLSR components")
        
        rmse_cv_array = []
        
        for i in tqdm(range(100), desc="CV optimization"):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
            plsr = PLSRegression(n_components=n_components, scale=False)
            
            rmse_train = []
            for train_idx, val_idx in skf.split(X_train, ce_binned):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                plsr.fit(X_fold_train, y_fold_train)
                y_pred = plsr.predict(X_fold_val)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                rmse_train.append(rmse)
            
            rmse_cv_array.append(np.mean(rmse_train))
        
        best_idx = np.argmin(rmse_cv_array)
        best_rmse = rmse_cv_array[best_idx]
        
        print(f"✓ Best random state: {best_idx} (CV RMSE: {best_rmse:.4f})")
        
        return best_idx
    
    def train_plsr_ensemble(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                           ce_binned, best_random_state, n_components=11):
        """
        Train 5 PLSR models using StratifiedKFold with best random state
        """
        print(f"\nTraining PLSR ensemble with {n_components} components...")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=best_random_state)
        
        pred_train_array = []
        pred_val_array = []
        pred_test_array = []
        
        fold = 1
        for train_idx, _ in skf.split(X_train, ce_binned):
            print(f"  Training fold {fold}/5...")
            
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            
            plsr = PLSRegression(n_components=n_components, scale=False)
            plsr.fit(X_fold_train, y_fold_train)
            
            # Make predictions
            y_pred_train = plsr.predict(X_train)
            pred_train_array.append(y_pred_train)
            
            y_pred_val = plsr.predict(X_val)
            pred_val_array.append(y_pred_val)
            
            y_pred_test = plsr.predict(X_test)
            pred_test_array.append(y_pred_test)
            
            # Save model
            model_path = self.model_dir / f'plsr_ce_strat_ncomp_{n_components}_fold_{fold}.sav'
            with open(model_path, 'wb') as f:
                pickle.dump(plsr, f)
            
            fold += 1
        
        print(f"✓ Trained and saved 5 PLSR models to {self.model_dir.relative_to(Path(__file__).resolve().parent)}/")
        
        # Save scaler
        scaler_path = self.model_dir / 'minmax_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved MinMaxScaler")
        
        # Calculate ensemble predictions
        y_pred_train_avg = np.mean(pred_train_array, axis=0).flatten()
        y_pred_val_avg = np.mean(pred_val_array, axis=0).flatten()
        y_pred_test_avg = np.mean(pred_test_array, axis=0).flatten()
        
        return y_pred_train_avg, y_pred_val_avg, y_pred_test_avg
    
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
            metrics_train, metrics_val, metrics_test: Dicts with 'rmse' keys
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
        ax.text(0.5, 0.98, 'Electrolytomics Dataset', transform=ax.transAxes,
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
        
        print(f"\n✓ Parity plot saved to: {output_path.relative_to(Path(__file__).resolve().parent)}")
    
    def run(self):
        """Complete training pipeline"""
        print("\n" + "="*70)
        print("CE PLSR Model Training - LUMO Chemprop Embeddings + PLSR Ensemble")
        print("="*70 + "\n")
        
        # Load Chemprop model
        print("Step 1: Loading LUMO Chemprop model")
        print("-" * 70)
        if not self.load_chemprop():
            return False
        
        # Load CE dataset
        print(f"\nStep 2: Loading CE dataset")
        print("-" * 70)
        print(f"Reading {self.ce_dataset_path.relative_to(Path(__file__).resolve().parent)}...")
        df = pd.read_csv(self.ce_dataset_path)
        print(f"✓ Loaded {len(df)} CE samples")
        
        # Extract features
        print(f"\nStep 3: Feature extraction")
        print("-" * 70)
        X_features = self.extract_features(df)
        
        # Combine with original data
        # First 2 columns are identifiers (we'll use indices)
        df_features = df.copy()
        feature_cols = [f'feat_{i}' for i in range(X_features.shape[1])]
        df_features_only = pd.DataFrame(X_features, columns=feature_cols)
        df_features = pd.concat([df_features, df_features_only], axis=1)
        
        # Create splits
        print(f"\nStep 4: Creating train/val/test splits")
        print("-" * 70)
        df_train, df_val, df_test = self.create_splits(df_features)
        
        # Prepare data
        print(f"\nStep 5: Data preparation")
        print("-" * 70)
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
            df_train, df_val, df_test
        )
        
        # Get CE_binned for stratification
        ce_binned_train = df_train['CE_binned'].values
        
        # Optimize CV
        print(f"\nStep 6: Hyperparameter optimization")
        print("-" * 70)
        best_random_state = self.optimize_cv_splits(
            X_train, y_train, ce_binned_train, n_components=11
        )
        
        # Train ensemble
        print(f"\nStep 7: Training PLSR ensemble")
        print("-" * 70)
        y_pred_train, y_pred_val, y_pred_test = self.train_plsr_ensemble(
            X_train, y_train, X_val, y_val, X_test, y_test,
            ce_binned_train, best_random_state, n_components=11
        )
        
        # Evaluate
        print(f"\nStep 8: Evaluation")
        print("-" * 70)
        metrics_train = self.evaluate(y_train, y_pred_train, "Train")
        metrics_val = self.evaluate(y_val, y_pred_val, "Validation")
        metrics_test = self.evaluate(y_test, y_pred_test, "Test")
        
        # Create parity plot
        print(f"\nGenerating parity plot...")
        plot_path = self.results_dir / 'training_parity_plot.png'
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
        print(f"  - plsr_ce_strat_ncomp_11_fold_[1-5].sav")
        print(f"  - minmax_scaler.pkl")
        print(f"  - training_metrics.csv")
        print(f"\nData splits saved to: {self.output_dir.relative_to(Path(__file__).resolve().parent)}/")
        print(f"  - train_features.csv")
        print(f"  - val_features.csv")
        print(f"  - test_features.csv")
        print(f"\nResults saved to: {self.results_dir.relative_to(Path(__file__).resolve().parent)}/")
        print(f"  - training_parity_plot.png")
        print("\n" + "="*70 + "\n")
        
        return True


def main():
    """Run CE model training"""
    trainer = CEModelTrainer()
    success = trainer.run()
    
    if not success:
        print("✗ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

