"""
Evaluate the trained LUMO Chemprop model on test set

Loads the trained model and runs evaluation that failed during training.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Patch torch.load for PyTorch 2.6
_orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Patch sklearn
from sklearn import metrics
_orig_mse = metrics.mean_squared_error

def patched_mse(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True):
    try:
        return _orig_mse(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput, squared=squared)
    except TypeError:
        result = _orig_mse(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
        return result if squared else result ** 0.5

metrics.mean_squared_error = patched_mse


def evaluate_model():
    """Load and evaluate the trained model"""
    print("="*60)
    print("Evaluating LUMO Chemprop Model")
    print("="*60 + "\n")
    
    # Paths - in this folder
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / 'lumo_models' / 'fold_0' / 'model_0' / 'model.pt'
    test_csv = script_dir / 'lumo_datasets' / 'lumo_test.csv'
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run pretrain_lumo_model.py first")
        return False
    
    if not test_csv.exists():
        print(f"ERROR: Test data not found at {test_csv}")
        return False
    
    print(f"Loading model from {model_path}...")
    
    try:
        from chemprop.utils import load_checkpoint
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint to get scaler
        checkpoint = torch.load(str(model_path), map_location=device)
        scaler = checkpoint['data_scaler']
        scaler_mean = scaler['means'][0]
        scaler_std = scaler['stds'][0]
        
        print(f"Data scaler: mean={scaler_mean:.4f}, std={scaler_std:.4f}")
        
        model = load_checkpoint(str(model_path), device=device)
        print(f"✓ Model loaded successfully\n")
        
        # Load test data
        print(f"Loading test data from {test_csv}...")
        test_df = pd.read_csv(test_csv)
        print(f"✓ Loaded {len(test_df)} test samples\n")
        
        # Make predictions in batches
        print("Running predictions on test set...")
        model.eval()
        
        preds = []
        batch_size = 50
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_df), batch_size), desc="Predicting"):
                batch_smiles = test_df['smiles'].iloc[i:i+batch_size].tolist()
                
                # Chemprop expects list of lists
                batch = [[s] for s in batch_smiles]
                
                # Get predictions (normalized)
                batch_preds = model(batch)
                preds.extend(batch_preds.detach().cpu().numpy().flatten().tolist())
        
        # Denormalize predictions: pred_original = pred_normalized * std + mean
        preds = np.array(preds)
        preds_denormalized = preds * scaler_std + scaler_mean
        
        targets = test_df['lumo'].values
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
        
        mae = mean_absolute_error(targets, preds_denormalized)
        r2 = r2_score(targets, preds_denormalized)
        rmse = np.sqrt(mean_squared_error(targets, preds_denormalized))
        
        print("\n" + "="*60)
        print("Test Set Results")
        print("="*60)
        print(f"MAE:  {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²:   {r2:.6f}")
        print("="*60)
        
        if r2 >= 0.95:
            print("\n✓ Model performance excellent (R² >= 0.95)")
        elif r2 >= 0.90:
            print("\n✓ Model performance good (R² >= 0.90)")
        else:
            print(f"\n⚠ Model performance below target (R² = {r2:.4f}, target ~0.98)")
        
        # Save predictions
        test_df['predicted_lumo'] = preds_denormalized
        test_df['error'] = preds_denormalized - targets
        output_file = script_dir / 'lumo_datasets' / 'lumo_test_predictions.csv'
        test_df.to_csv(output_file, index=False)
        print(f"\n✓ Predictions saved to: lumo_datasets/{output_file.name}\n")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = evaluate_model()
    sys.exit(0 if success else 1)

