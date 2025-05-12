# data_loader.py
from datasets import load_dataset
from config import SUBSET

# ─────────────────────────── DATA LOADING ───────────────────────────────
print("Loading FEVER train split …")
full_ds = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
init_ds = full_ds.select(range(min(SUBSET, len(full_ds))))

# Deduplicate claims within the sampled slice
seen, unique_indices = set(), []
for idx, ex in enumerate(init_ds):
    claim = ex["claim"].strip()
    if claim not in seen:
        seen.add(claim)
        unique_indices.append(idx)

ds = init_ds.select(unique_indices)
print(f"Using {len(ds)} unique examples (from first {len(init_ds)})\n")
