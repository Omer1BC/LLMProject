import torch
from transformers import logging as transformers_logging

# ─────────────────────────── GLOBAL CONFIG ──────────────────────────────
transformers_logging.set_verbosity_error()

MODEL_REPO_DEFAULT = "tiiuae/falcon-7b-instruct"
EXPLAINER_REPO     = "tiiuae/falcon-7b-instruct"      # swap out freely if desired
LABELS             = ["FAKE", "TRUE"]                # ← binary task
SUBSET             = 1500                              # demo-friendly slice
SEARCH_SNIPPETS    = 3
MAX_GEN_TOKENS     = 1024
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
