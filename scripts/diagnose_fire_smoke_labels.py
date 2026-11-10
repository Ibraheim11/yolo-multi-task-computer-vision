"""Colab/local CLI: offline label diagnosis; repair requires an explicit reviewed plan."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.label_diagnostics import create_reviewed_copy, diagnose_labels, write_diagnosis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Actual exported dataset root")
    parser.add_argument("--report", type=Path, help="Existing dataset_audit/report.json (optional with dataset)")
    parser.add_argument("--output", type=Path, required=True, help="Diagnostic output directory outside original export")
    parser.add_argument("--reviewed-plan", type=Path, help="Explicit human-reviewed line replacement JSON; no guessed conversion")
    parser.add_argument("--clean-copy", type=Path, help="New destination outside original export; required with reviewed-plan")
    args = parser.parse_args()
    if bool(args.reviewed_plan) != bool(args.clean_copy) or (args.reviewed_plan and not args.dataset):
        parser.error("Repair requires --dataset, --reviewed-plan and --clean-copy together")
    diagnosis = diagnose_labels(args.dataset, args.report)
    print(write_diagnosis(diagnosis, args.output))
    if args.reviewed_plan:
        manifest = create_reviewed_copy(args.dataset, args.clean_copy, args.reviewed_plan)
        print(f"Created reviewed copy. Audit ready: {manifest['ready']}. Check repair_manifest.json and repair_audit.json.")
    print("Diagnosis only unless an explicit reviewed plan was supplied. No training or inference is performed.")


if __name__ == "__main__":
    main()
