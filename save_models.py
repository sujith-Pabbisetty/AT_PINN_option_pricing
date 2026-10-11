#!/usr/bin/env python
"""
save_models.py
--------------
Run this script AFTER 02_at_pinn_full_paper.py (or the notebook) has finished
training.  It picks up the trained objects from the caller's namespace if
exec()'d inline, or re-executes the training script and then saves weights.

Usage (standalone):
    python save_models.py

Usage (after running the notebook in the same process):
    exec(open("save_models.py").read())
"""

import os
import sys
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_if_available(obj, path, name):
    """Save a PyTorch module's state_dict if it exists."""
    try:
        if obj is None:
            print(f"  [SKIP] {name}: object is None, nothing to save.")
            return 0
        state = obj.state_dict() if hasattr(obj, "state_dict") else obj
        torch.save(state, path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  [OK]  {name} -> {path}  ({size_kb:.1f} KB)")
        return 1
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        return 0


def save_list(model_list, path, name):
    """Save a list/ensemble of modules as a list of state_dicts."""
    try:
        if not model_list:
            print(f"  [SKIP] {name}: list is empty, nothing to save.")
            return 0
        state_dicts = [m.state_dict() for m in model_list]
        torch.save(state_dicts, path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  [OK]  {name} ({len(model_list)} members) -> {path}  ({size_kb:.1f} KB)")
        return 1
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        return 0


# ── Try to pick up objects already in the caller's namespace ─────────────────
_g = globals()
ens_eu = _g.get("ens_eu", None)
ens_am = _g.get("ens_am", None)
m1_eu  = _g.get("m1_eu",  None)
m1_am  = _g.get("m1_am",  None)

# ── If not found, run the training script first ──────────────────────────────
if all(v is None for v in [ens_eu, ens_am, m1_eu, m1_am]):
    print("No trained models found in current namespace.")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_at_pinn_full_paper.py")
    if not os.path.exists(script):
        print(f"Training script not found at: {script}")
        print("Please run the training script first, then call save_models.py.")
        sys.exit(1)
    print(f"Running training script: {script}")
    print("(This may take a while depending on MODE setting...)\n")
    _ns = {}
    exec(open(script).read(), _ns)   # noqa: S102
    ens_eu = _ns.get("ens_eu")
    ens_am = _ns.get("ens_am")
    m1_eu  = _ns.get("m1_eu")
    m1_am  = _ns.get("m1_am")

# ── Save weights ─────────────────────────────────────────────────────────────
print("\nSaving model weights to results/")
saved = 0
saved += save_list(ens_eu,  os.path.join(RESULTS_DIR, "pinn_european_ensemble.pt"), "ens_eu  (European ensemble)")
saved += save_list(ens_am,  os.path.join(RESULTS_DIR, "pinn_american_ensemble.pt"), "ens_am  (American ensemble)")
saved += save_if_available(m1_eu, os.path.join(RESULTS_DIR, "pinn_european_stage1.pt"), "m1_eu   (European stage-1)")
saved += save_if_available(m1_am, os.path.join(RESULTS_DIR, "pinn_american_stage1.pt"), "m1_am   (American stage-1)")

print(f"\nDone. {saved}/4 model objects saved to '{RESULTS_DIR}'.")
if saved < 4:
    print("Some models were missing — re-run the training script and try again.")