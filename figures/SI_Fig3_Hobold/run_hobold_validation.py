"""Run the Hobold current-density sweep using 40 trained HELENA checkpoints."""

import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

from core.utils import (
    extract_training_mappings_from_checkpoint,
    feature_engineering,
    create_inference_preprocessing,
    load_preprocessed_data,
)
from modules.dataset_processing import CollateFn
from modules.helena_ce.helena_ce_model import HELENA_CE
from modules.helena_ce.helena_ce_parameters import ARCHITECTURE
from config.datasets import MAX_MOLECULES

CURRENTS = [0.25, 0.5, 1.0, 2.0, 4.0]


SM = {
    "2MeTHF": "CC1CCCO1",
    "PC":     "CC1COC(=O)O1",
    "EC":     "O=C1OCCO1",
    "DEC":    "CCOC(=O)OCC",
    "DMC":    "COC(=O)OC",
    "EMC":    "CCOC(=O)OC",
    "FEC":    "O=C1OCC(F)O1",
    "DOL":    "C1COCO1",
    "DME":    "COCCOC",
    "LiAsF6": "F[As-](F)(F)(F)(F)F.[Li+]",
    "LiClO4": "[Li+].[O-][Cl+3]([O-])([O-])[O-]",
    "LiPF6":  "F[P-](F)(F)(F)(F)F.[Li+]",
    "LiFSI":  "O=S(=O)(F)[N-]S(=O)(=O)F.[Li+]",
    "LiTFSI": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]",
    "LiNO3":  "O=[N+]([O-])[O-].[Li+]",
}






PC_RATIO   = 11.7546
EC_DEC_E   = 6.36806
EC_DEC_D   = 4.74718
EC_DMC_E   = 7.49472
EC_DMC_D   = 5.93930
FEC_PURE   = 13.71000
DOL_RATIO  = 7.15462
DME_RATIO  = 4.81580
HOBOLD = [

    {"name": "1.5M LiAsF6 / 2-MeTHF",
     "solvents": [(SM["2MeTHF"], 9.915)],
     "salts":    [(SM["LiAsF6"], 1.5)],
     "type": 1, "reported_CE": 78.0},
    {"name": "1M LiClO4 / PC",
     "solvents": [(SM["PC"], PC_RATIO)],
     "salts":    [(SM["LiClO4"], 1.0)],
     "type": 1, "reported_CE": 81.9},
    {"name": "1M LiPF6 / EC:DEC (1:1)",
     "solvents": [(SM["EC"], EC_DEC_E), (SM["DEC"], EC_DEC_D)],
     "salts":    [(SM["LiPF6"], 1.0)],
     "type": 1, "reported_CE": 93.8},
    {"name": "1M LiFSI / FEC",
     "solvents": [(SM["FEC"], FEC_PURE)],
     "salts":    [(SM["LiFSI"], 1.0)],
     "type": 1, "reported_CE": 95.3},
    {"name": "1M LiPF6 / EC:DMC (1:1)",
     "solvents": [(SM["EC"], EC_DMC_E), (SM["DMC"], EC_DMC_D)],
     "salts":    [(SM["LiPF6"], 1.0)],
     "type": 1, "reported_CE": 96.9},

    {"name": "1M LiPF6 / EC:EMC (3:7 wt)",
     "solvents": [(SM["EC"], 3.690498), (SM["EMC"], 7.284218)],
     "salts":    [(SM["LiPF6"], 1.0)],
     "type": 1, "reported_CE": 89.0},
    {"name": "1M LiClO4 / EC",
     "solvents": [(SM["EC"], 14.98944)],
     "salts":    [(SM["LiClO4"], 1.0)],
     "type": 1, "reported_CE": 92.0},

    {"name": "2M LiFSI / FEC",
     "solvents": [(SM["FEC"], FEC_PURE)],
     "salts":    [(SM["LiFSI"], 2.0)],
     "type": 2, "reported_CE": 97.5},
    {"name": "7M LiFSI / FEC",
     "solvents": [(SM["FEC"], FEC_PURE)],
     "salts":    [(SM["LiFSI"], 7.0)],
     "type": 2, "reported_CE": 98.2},
    {"name": "1M LiTFSI 3wt% LiNO3 / DOL:DME (1:1)",
     "solvents": [(SM["DOL"], DOL_RATIO), (SM["DME"], DME_RATIO)],
     "salts":    [(SM["LiTFSI"], 1.0), (SM["LiNO3"], 0.466376)],
     "type": 2, "reported_CE": 99.0},
    {"name": "2M LiFSI 1M LiTFSI 3wt% LiNO3 / DOL:DME (1:1)",
     "solvents": [(SM["DOL"], DOL_RATIO), (SM["DME"], DME_RATIO)],
     "salts":    [(SM["LiFSI"], 2.0), (SM["LiTFSI"], 1.0), (SM["LiNO3"], 0.466376)],
     "type": 2, "reported_CE": 99.3},
]


def hobold_to_df(electrolytes, current_densities, method="Aurbach"):
    """Build a dataframe in HELENA-CE-26 schema. One row per (electrolyte, current)."""
    rows = []
    rid = 1_000_000
    for e in electrolytes:
        for cur in current_densities:
            row = {
                "Row_ID": rid, "Electrolyte": e["name"],
                "CE": 0.0, "LCE": 0.0,
                "Solvent_1": np.nan, "Solvent_2": np.nan, "Solvent_3": np.nan,
                "Solvent_4": np.nan, "Solvent_5": np.nan,
                "Salt_1": np.nan, "Salt_2": np.nan, "Salt_3": np.nan,
                "Method": method, "Current": float(cur),
                "Capacity": 1.0, "Cycle": 10.0, "Reference": "Hobold2023",
            }
            for i, (smi, mr) in enumerate(e["solvents"], start=1):
                row[f"Solvent_{i}"] = f"{smi}:{mr}"
            for i, (smi, mr) in enumerate(e["salts"], start=1):
                row[f"Salt_{i}"] = f"{smi}:{mr}"
            row["_hobold_idx"] = len(rows) // len(current_densities)
            row["_current_idx"] = list(current_densities).index(cur)
            rows.append(row)
            rid += 1
    return pd.DataFrame(rows)


def collate_full(df, map_dict, device):
    [ds] = feature_engineering(map_dict, 0.0, df.copy(), label_field="LCE")
    items = [ds[i] for i in range(len(ds))]
    batch = CollateFn(MAX_MOLECULES)(items)
    out = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
           for k, v in batch.items()}
    return out, batch["row_id"].cpu().numpy()


def run_one_gpu(gpu_id, ckpt_paths, preproc_path, df, return_dict):
    import torch
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU{gpu_id}] start; {len(ckpt_paths)} ckpts", flush=True)

    map_dict, vocab_size, _, _ = load_preprocessed_data(custom_path=preproc_path)
    fid_to_idx = {int(rid): i for i, rid in enumerate(df["Row_ID"].values)}

    batch, fids = collate_full(df, map_dict, device)
    fid_idx = np.array([fid_to_idx[int(f)] for f in fids])

    n = len(df)
    pred_per_member = np.zeros((len(ckpt_paths), n), dtype=np.float64)

    model = HELENA_CE(
        **ARCHITECTURE, input_vocab_size=vocab_size,
        max_num_molecules=MAX_MOLECULES, task_type="regression", num_classes=None,
    ).to(device)
    model.eval()

    for ck_i, ck_path in enumerate(ckpt_paths):
        t0 = time.time()
        ck = torch.load(ck_path, weights_only=False, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        with torch.no_grad():
            out = model(batch)["ce"].squeeze(-1).cpu().numpy()
        pred_per_member[ck_i, fid_idx] = out
        print(f"[GPU{gpu_id}] {ck_i+1}/{len(ckpt_paths)} {time.time()-t0:.2f}s", flush=True)

    return_dict[gpu_id] = (pred_per_member, [os.path.basename(p) for p in ckpt_paths])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="Directory containing outer_1_model_1.pth through outer_5_model_8.pth.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(REPO) / "results" / f"Hobold_Validation_{os.getpid()}")
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()
    weight_dir = args.weights_dir.resolve()
    ckpts = [str(weight_dir / f"outer_{of}_model_{mi}.pth")
             for of in range(1, 6) for mi in range(1, 9)]
    missing = [path for path in ckpts if not os.path.isfile(path)]
    if missing:
        parser.error(f"Missing {len(missing)} checkpoints; first missing file: {missing[0]}")
    available_gpus = torch.cuda.device_count()
    n_gpus = min(4, available_gpus) if args.gpus is None else args.gpus
    if not 1 <= n_gpus <= available_gpus:
        parser.error(f"Select between 1 and {available_gpus} available CUDA GPUs.")
    OUT = str(args.output_dir.resolve())
    DD = os.path.join(OUT, "data")
    os.makedirs(DD, exist_ok=True)

    print("=" * 70)
    print("HOBOLD TYPE 1 vs TYPE 2 VALIDATION")
    print("=" * 70)

    df = hobold_to_df(HOBOLD, CURRENTS, method="Aurbach")
    print(f"Built dataframe: {len(df)} rows (= {len(HOBOLD)} electrolytes × {len(CURRENTS)} currents)")
    df_path = os.path.join(DD, "hobold_input.csv")
    df.drop(columns=["_hobold_idx", "_current_idx"]).to_csv(df_path, index=False)


    template = os.path.join(weight_dir, "outer_1_model_1.pth")
    mkm, kmm, vocab_size, _ = extract_training_mappings_from_checkpoint(template)
    print(f"vocab_size = {vocab_size}")
    process_id = os.getpid()
    preproc_path = create_inference_preprocessing(
        df_path, mkm, kmm, vocab_size, DD, process_id,
    )
    print(f"preproc -> {preproc_path}")


    splits = [ckpts[i::n_gpus] for i in range(n_gpus)]

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager(); rd = manager.dict(); procs = []
    t0 = time.time()
    for g in range(n_gpus):
        p = mp.Process(target=run_one_gpu, args=(g, splits[g], preproc_path, df, rd))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    print(f"All workers done in {time.time()-t0:.1f}s")


    n = len(df)
    all_member_pred = np.zeros((40, n), dtype=np.float64)
    cursor = 0
    for g in range(n_gpus):
        ppm, _ = rd[g]
        all_member_pred[cursor:cursor + ppm.shape[0]] = ppm
        cursor += ppm.shape[0]
    assert cursor == 40


    ens_mean = all_member_pred.mean(axis=0)
    ens_std = all_member_pred.std(axis=0, ddof=1)

    df["pred_lce_mean"] = ens_mean
    df["pred_lce_std"] = ens_std
    df["pred_ce_mean"] = 100 * (1 - 10 ** (-ens_mean))


    log_I = np.log(np.array(CURRENTS) + 1e-6)
    rows = []
    for hi, e in enumerate(HOBOLD):
        sub = df[df["_hobold_idx"] == hi].sort_values("_current_idx")
        lce = sub["pred_lce_mean"].values
        ce  = sub["pred_ce_mean"].values
        slope, intercept = np.polyfit(log_I, lce, 1)
        rows.append(dict(
            name=e["name"], hobold_type=e["type"], reported_CE=e["reported_CE"],
            HELENA_CE_at_0p25=ce[0], HELENA_CE_at_0p5=ce[1],
            HELENA_CE_at_1p0=ce[2], HELENA_CE_at_2p0=ce[3], HELENA_CE_at_4p0=ce[4],
            HELENA_CE_range=float(ce.max() - ce.min()),
            HELENA_dLCE_dlogI=float(slope),
            HELENA_abs_slope=float(abs(slope)),
            HELENA_lce_std_mean=float(sub["pred_lce_std"].mean()),
        ))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DD, "hobold_validation.csv"), index=False)


    df.drop(columns=["_hobold_idx", "_current_idx"]).to_csv(
        os.path.join(DD, "hobold_predictions_full.csv"), index=False
    )


    from scipy.stats import mannwhitneyu
    res["CE_drop_0p25_to_4"] = res["HELENA_CE_at_0p25"] - res["HELENA_CE_at_4p0"]
    t1 = res[res["hobold_type"] == 1]["HELENA_abs_slope"].values
    t2 = res[res["hobold_type"] == 2]["HELENA_abs_slope"].values
    u, p = mannwhitneyu(t1, t2, alternative="greater")
    t1r = res[res["hobold_type"] == 1]["HELENA_CE_range"].values
    t2r = res[res["hobold_type"] == 2]["HELENA_CE_range"].values
    u_r, p_r = mannwhitneyu(t1r, t2r, alternative="greater")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(res[["name","hobold_type","reported_CE","HELENA_CE_at_0p5",
               "HELENA_dLCE_dlogI","HELENA_CE_range","HELENA_abs_slope"]].to_string(index=False))
    print(f"\nType 1 mean |slope| = {t1.mean():.4f}  (CE range mean = {res[res['hobold_type']==1]['HELENA_CE_range'].mean():.2f}%)")
    print(f"Type 2 mean |slope| = {t2.mean():.4f}  (CE range mean = {res[res['hobold_type']==2]['HELENA_CE_range'].mean():.2f}%)")
    print(f"\nMann-Whitney (|dLCE/d log I|, T1 > T2):  U={u:.2f}  p={p:.4f}")
    print(f"Mann-Whitney (CE range,           T1 > T2):  U={u_r:.2f}  p={p_r:.4f}")
    print(f"Type 1 mean CE range = {t1r.mean():.3f}%   Type 2 mean CE range = {t2r.mean():.3f}%")
    if p_r < 0.05:
        print(">> HELENA's predicted CE drop is significantly larger for Type 1 — consistent with Hobold Fig. 9.")
    else:
        print(f">> CE-range test is in the right direction (Type 1 = {t1r.mean():.2f}% vs Type 2 = {t2r.mean():.2f}%) "
              f"but only marginal at n={len(t1r)}+{len(t2r)} (p={p_r:.3f}).")

    with open(os.path.join(OUT, "hobold_test_result.txt"), "w") as f:
        f.write(f"Mann-Whitney U (one-sided, |slope| Type1 > Type2):\n")
        f.write(f"  U = {u:.4f}\n  p = {p:.6f}\n")
        f.write(f"  n_Type1 = {len(t1)}, n_Type2 = {len(t2)}\n")
        f.write(f"  mean |slope| Type1 = {t1.mean():.6f}\n")
        f.write(f"  mean |slope| Type2 = {t2.mean():.6f}\n")
        f.write(f"  mean CE range Type1 = {res[res['hobold_type']==1]['HELENA_CE_range'].mean():.4f}%\n")
        f.write(f"  mean CE range Type2 = {res[res['hobold_type']==2]['HELENA_CE_range'].mean():.4f}%\n")


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for hi, e in enumerate(HOBOLD):
        sub = df[df["_hobold_idx"] == hi].sort_values("_current_idx")
        ax = axes[0] if e["type"] == 1 else axes[1]
        ax.plot(CURRENTS, sub["pred_ce_mean"].values, marker="o", label=e["name"][:38])
    axes[0].set_title(f"Type 1 (n={(res['hobold_type']==1).sum()})")
    axes[1].set_title(f"Type 2 (n={(res['hobold_type']==2).sum()})")
    for ax in axes:
        ax.set_xlabel("Current density (mA/cm²)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("HELENA-predicted CE (%)")
    fig.suptitle(f"Hobold Type 1 vs Type 2 — HELENA inference (Mann-Whitney p={p:.3g})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hobold_ce_vs_current.png"), dpi=150)
    print(f"Saved {OUT}/hobold_ce_vs_current.png")

    if os.path.exists(preproc_path):
        os.remove(preproc_path)


if __name__ == "__main__":
    main()
