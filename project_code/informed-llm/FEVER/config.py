# config.py
import torch
from transformers import logging as transformers_logging

# ─────────────────────────── GLOBAL CONFIG ──────────────────────────────
transformers_logging.set_verbosity_error()

MODEL_REPO_DEFAULT = "tiiuae/falcon-7b-instruct"
EXPLAINER_REPO = "tiiuae/falcon-7b-instruct"  # could swap out for a slimmer model
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
SUBSET = 30  # keep runtime demo-friendly
SEARCH_SNIPPETS = 3
MAX_GEN_TOKENS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"