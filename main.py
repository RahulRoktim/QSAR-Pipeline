"""AutoQSAR pipeline entry point.

Every setting that affects a result is a command-line argument with a recorded
default, so a run can be reproduced from its manifest alone:

    python main.py --target BRAF --activity-type IC50
    python main.py --target PI3K --target-chembl-id CHEMBL3145 --cv-folds 10
    python main.py --target BRAF --dry-run        # resolve + validate, run nothing

Overrides are applied to the ``config`` module before any pipeline stage is
imported, so the existing ``from config import ...`` bindings pick them up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AutoQSAR: ChEMBL download -> descriptors -> model -> report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("dataset")
    data.add_argument("--target", default=config.TARGET,
                      help="Target label used for naming and identity checks.")
    data.add_argument("--target-chembl-id", default=config.TARGET_CHEMBL_ID,
                      help="Explicit ChEMBL target id. Verified against --target "
                           "before any download. Omit to resolve automatically.")
    data.add_argument("--activity-type", default=config.ACTIVITY_TYPE,
                      help="Bioactivity endpoint to retrieve (IC50, Ki, EC50).")
    data.add_argument("--min-compounds", type=int, default=config.MIN_COMPOUNDS,
                      help="Abort if fewer usable compounds survive preprocessing.")

    model = parser.add_argument_group("modelling")
    model.add_argument("--test-size", type=float, default=config.TEST_SIZE,
                       help="Hold-out fraction.")
    model.add_argument("--cv-folds", type=int, default=config.CV_FOLDS,
                       help="Cross-validation folds used for feature selection.")
    model.add_argument("--random-state", type=int, default=config.RANDOM_STATE,
                       help="Seed for splitting, selection and every model.")
    model.add_argument("--no-compare-models", dest="compare_models",
                       action="store_false", default=config.COMPARE_MODELS,
                       help="Train only the default estimator.")
    model.add_argument("--hyperparameter-search", action="store_true",
                       default=config.HYPERPARAMETER_SEARCH,
                       help="Enable hyperparameter search (slow).")

    validation = parser.add_argument_group("validation")
    validation.add_argument("--no-y-randomization", dest="run_y_randomization",
                            action="store_false", default=config.RUN_Y_RANDOMIZATION,
                            help="Skip y-scrambling. Not recommended: it is the "
                                 "control that shows the model is not fitting noise.")
    validation.add_argument("--y-randomization-runs", type=int,
                            default=config.Y_RANDOMIZATION_RUNS,
                            help="Number of y-scrambling repeats.")
    validation.add_argument("--shap", dest="run_shap", action="store_true",
                            default=config.RUN_SHAP, help="Compute SHAP explanations.")

    run = parser.add_argument_group("run control")
    run.add_argument("--skip-download", action="store_true",
                     help="Reuse the existing raw dataset instead of querying ChEMBL.")
    run.add_argument("--manifest", default="outputs/run_manifest.json",
                     help="Where to write the reproducibility manifest.")
    run.add_argument("--dry-run", action="store_true",
                     help="Resolve and validate the target, print the plan, exit.")

    return parser


SETTING_KEYS = {
    "target": "TARGET",
    "target_chembl_id": "TARGET_CHEMBL_ID",
    "activity_type": "ACTIVITY_TYPE",
    "min_compounds": "MIN_COMPOUNDS",
    "test_size": "TEST_SIZE",
    "cv_folds": "CV_FOLDS",
    "random_state": "RANDOM_STATE",
    "compare_models": "COMPARE_MODELS",
    "hyperparameter_search": "HYPERPARAMETER_SEARCH",
    "run_y_randomization": "RUN_Y_RANDOMIZATION",
    "y_randomization_runs": "Y_RANDOMIZATION_RUNS",
    "run_shap": "RUN_SHAP",
}


def apply_overrides(args: argparse.Namespace, module=config) -> dict:
    """Push CLI values onto the config module and return the resolved settings."""
    settings = {}

    for arg_name, config_name in SETTING_KEYS.items():
        value = getattr(args, arg_name)
        setattr(module, config_name, value)
        settings[config_name] = value

    return settings


def validate_settings(settings: dict) -> None:
    """Reject combinations that cannot produce a defensible model."""
    if not 0.0 < settings["TEST_SIZE"] < 1.0:
        raise ValueError(f"--test-size must be in (0, 1), got {settings['TEST_SIZE']}")

    if settings["CV_FOLDS"] < 2:
        raise ValueError(f"--cv-folds must be >= 2, got {settings['CV_FOLDS']}")

    if settings["MIN_COMPOUNDS"] < 1:
        raise ValueError("--min-compounds must be >= 1")

    if settings["Y_RANDOMIZATION_RUNS"] < 1 and settings["RUN_Y_RANDOMIZATION"]:
        raise ValueError("--y-randomization-runs must be >= 1 when enabled")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = apply_overrides(args)
    validate_settings(settings)

    print("\n" + "=" * 54)
    print("  AutoQSAR Pipeline")
    print("=" * 54)
    for key in sorted(settings):
        print(f"  {key:<24} {settings[key]}")
    print("=" * 54)

    from src.chembl_downloader import ChEMBLDownloader
    from src.provenance import build_manifest, write_manifest

    downloader = ChEMBLDownloader()
    resolved = downloader.resolve_target()

    print(f"\n  Resolved target : {resolved.chembl_id}  {resolved.pref_name}")
    print(f"  Organism        : {resolved.organism}")
    print(f"  Identity source : {resolved.source}\n")

    if args.dry_run:
        print("Dry run: target resolved and validated, no data downloaded.")
        return 0

    if not args.skip_download:
        downloader.run()

    from src.preprocess import preprocess
    from src.descriptors import calculate_descriptors
    from src.feature_selection import feature_selection
    from src.train import train_model
    from src.evaluate import evaluate_model

    preprocess()
    calculate_descriptors()
    feature_selection()
    train_model()
    evaluate_model()

    manifest = build_manifest(
        settings=settings,
        resolved_target=resolved.as_dict(),
        datasets={
            "raw": config.RAW_DATASET,
            "processed": config.PROCESSED_DATASET,
            "descriptors": config.DESCRIPTOR_DATASET,
            "selected": config.SELECTED_DATASET,
        },
        root=ROOT,
    )
    path = write_manifest(manifest, args.manifest)

    print("\n" + "=" * 54)
    print("  AutoQSAR Pipeline complete")
    print(f"  Manifest: {path}")
    print("=" * 54)
    return 0


if __name__ == "__main__":
    sys.exit(main())
