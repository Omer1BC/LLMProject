"""Build a small Fake-vs-True news dataset for benchmarking."""
from pathlib import Path

import pandas as pd
from datasets import Dataset

from config import SUBSET

DATA_DIR  = Path("./fake-news-data")
FAKE_CSV  = DATA_DIR / "Fake.csv"
TRUE_CSV  = DATA_DIR / "True.csv"

print("Loading Fake and True news CSV files …")
if not FAKE_CSV.exists() or not TRUE_CSV.exists():
    raise FileNotFoundError(
        f"Expected Fake.csv / True.csv in {DATA_DIR} – files missing."
    )

fake_df = pd.read_csv(FAKE_CSV)
true_df = pd.read_csv(TRUE_CSV)

fake_df["label"] = "FAKE"
true_df["label"] = "TRUE"

df = pd.concat([fake_df, true_df], ignore_index=True)

# Use the headline as the claim fed to the pipelines
df["claim"] = df["title"].astype(str).str.strip()

ds_full = Dataset.from_pandas(df[["claim", "label"]]).shuffle(seed=42)
ds      = ds_full.select(range(min(SUBSET, len(ds_full))))

print(f"Using {len(ds)} examples (subset of {len(ds_full)})\n")
