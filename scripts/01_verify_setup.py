#!/usr/bin/env python3
"""
Week 1 - verify the environment before touching anything else.

Run this first. It checks imports, GPU availability, and whether the dataset is reachable,
so that a missing dependency surfaces here rather than thirty minutes into a training run.

    python scripts/01_verify_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check_imports() -> bool:
    print("\n--- Core dependencies ---")
    ok = True
    for name in ["torch", "torchvision", "numpy", "pandas", "sklearn", "yaml",
                 "matplotlib", "tqdm", "PIL"]:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"{OK} {name:<14} {version}")
        except ImportError:
            print(f"{FAIL} {name:<14} not installed")
            ok = False
    return ok


def check_torch() -> None:
    print("\n--- Compute ---")
    try:
        import torch
    except ImportError:
        print(f"{FAIL} torch missing, skipping")
        return

    if torch.cuda.is_available():
        print(f"{OK} CUDA available: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{OK} GPU memory: {total:.1f} GB")
        if total < 6:
            print(f"{WARN} Under 6 GB. Reduce batch_size in configs/default.yaml if you hit OOM.")
    else:
        print(f"{WARN} No CUDA. Training will run on CPU and will be very slow.")
        print("       Consider Google Colab or Kaggle for the training runs.")


def check_datasets_lib() -> bool:
    print("\n--- Dataset library ---")
    try:
        import datasets
        print(f"{OK} datasets {datasets.__version__}")
        return True
    except ImportError:
        print(f"{FAIL} 'datasets' not installed  ->  pip install datasets")
        return False


# Published per-centre training counts, from the dataset card. Used as a sanity check:
# if these do not match, the mirror has changed and every later number is suspect.
EXPECTED_TRAIN = {0: 9930, 1: 3163, 2: 2691, 3: 1807, 4: 655, 5: 351}


def check_dataset() -> bool:
    print("\n--- Dataset ---")
    print("      First run downloads ~144 MB and may take a few minutes.")
    try:
        from src.data.loader import load_center_hf
        ds = load_center_hf(center=0, train=True)
    except Exception as exc:
        print(f"{FAIL} Could not load centre 0: {exc}")
        return False

    n = len(ds)
    expected = EXPECTED_TRAIN[0]
    if n == expected:
        print(f"{OK} Centre 0 loaded: {n:,} training samples (matches published count)")
    else:
        print(f"{WARN} Centre 0 has {n:,} samples, expected {expected:,}")
        print("       The mirror may have changed - verify before trusting results.")

    cols = getattr(ds, "column_names", [])
    if set(cols) >= {"image", "center", "label"}:
        print(f"{OK} Columns present: {cols}")
        return True
    print(f"{FAIL} Unexpected columns: {cols}")
    return False


def main() -> int:
    print("=" * 62)
    print("SETUP VERIFICATION")
    print("=" * 62)

    imports_ok = check_imports()
    check_torch()
    lib_ok = check_datasets_lib()
    data_ok = check_dataset() if lib_ok else False

    print("\n" + "=" * 62)
    if imports_ok and lib_ok and data_ok:
        print("All checks passed. Next: python scripts/02_explore_data.py")
        return 0
    if imports_ok and not data_ok:
        print("Dependencies fine; dataset not ready yet.")
        print("This is expected on day 1 - dataset access takes time to arrange.")
        return 1
    print("Install the missing dependencies above, then re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
