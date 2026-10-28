"""Run the ablation variants enabled in selected_configs."""

import regr_ablation
import warnings

warnings.filterwarnings("ignore")


def run_selected_ablations():
    """Run selected ablation studies based on configuration."""

    selected_configs = {
        "no_moiety_encoder": True,
        "no_molar_ratio": True,
        "no_salt_group": True,
        "no_solvent_group_attention": True,
        "no_formulation_attention": True,
        "no_current": True,
        "no_method": True,
    }

    original_configs = regr_ablation.ABLATION_CONFIGS
    selected_ablations = {
        k: original_configs[k]
        for k, v in selected_configs.items()
        if v and k in original_configs
    }

    if not selected_ablations:
        print(
            "No ablation configurations were selected. Please enable at least one configuration."
        )
        return

    print(f"\nRunning {len(selected_ablations)} selected ablation studies:")
    for key, config in selected_ablations.items():
        print(f"  - {config['name']}: {config['description']}")

    print(f"\nConfiguration:")
    print(f"  MASTER_SEED: {regr_ablation.MASTER_SEED}")
    print(f"  N_OUTER_FOLDS: {regr_ablation.N_OUTER_FOLDS}")
    print(f"  N_ENSEMBLE: {regr_ablation.N_ENSEMBLE}")
    print(
        f"  Total models per ablation: {regr_ablation.N_OUTER_FOLDS * regr_ablation.N_ENSEMBLE}"
    )

    regr_ablation.ABLATION_CONFIGS = selected_ablations

    regr_ablation.main()


if __name__ == "__main__":
    run_selected_ablations()
