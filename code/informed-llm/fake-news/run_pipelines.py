from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO, SEARCH_SNIPPETS
from data_loader import ds
from model import Model, ModelPipeline, AncCtx
from web_scraper import web_search

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────

def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]

def _add_web(ctx: AncCtx, claim: str) -> str:
    """Attach a single DuckDuckGo snippet block to the claim (cached per run)."""
    if "_web_ctx" not in ctx:
        ctx["_web_ctx"] = web_search(claim, max_results=SEARCH_SNIPPETS)
    return f"{claim}\n\nContext:\n{ctx['_web_ctx']}"

# ───────────────────────────────────────────────
# Debate Role Prompts
# ───────────────────────────────────────────────

_DEBATER_ROLES = {
    "FAKE": (
        "You are a fact-checking debater. Argue that the claim is FAKE. "
        "Explain in 1–3 paragraphs why the headline is fabricated or misleading."
    ),
    "TRUE": (
        "You are a fact-checking debater. Argue that the claim is TRUE. "
        "Explain in 1–3 paragraphs with evidence why the headline is accurate."
    ),
}

# ───────────────────────────────────────────────
# Model Definitions
# ───────────────────────────────────────────────

BASE = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    enforce_labels=True,
    instructions="Label the claim as FAKE or TRUE.",
    name="base-clf",
)

DEB_FAKE = Model(MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["FAKE"], enforce_labels=False, name="debater-FAKE")
DEB_TRUE = Model(MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["TRUE"], enforce_labels=False, name="debater-TRUE")

JUDGE = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="You are an impartial fact-checking judge. Choose FAKE or TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n"
        f"FAKE arguments:\n{ctx.get('Model(debater-FAKE)','')}\n\n"
        f"TRUE arguments:\n{ctx.get('Model(debater-TRUE)','')}\n\n"
        "Label:"
    ),
    name="judge",
)

RESP1_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-FAKE",
    instructions="Respond defending FAKE in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\nFAKE:\n{ctx.get('Model(debater-FAKE)','')}\n\nTRUE:\n{ctx.get('Model(debater-TRUE)','')}\n"
    ),
)

RESP1_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-TRUE",
    instructions="Respond defending TRUE in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\nFAKE:\n{ctx.get('Model(debater-FAKE)','')}\n\nTRUE:\n{ctx.get('Model(debater-TRUE)','')}\n"
    ),
)

RESP2_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-FAKE",
    instructions="Second response for FAKE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\nResp1 FAKE:\n{ctx.get('Model(response1-FAKE)','')}\n\nResp1 TRUE:\n{ctx.get('Model(response1-TRUE)','')}\n"
    ),
)

RESP2_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-TRUE",
    instructions="Second response for TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\nResp1 FAKE:\n{ctx.get('Model(response1-FAKE)','')}\n\nResp1 TRUE:\n{ctx.get('Model(response1-TRUE)','')}\n"
    ),
)

CLOS_FAKE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-FAKE",
    instructions="Closing remarks for FAKE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n{ctx.get('Model(response2-FAKE)','')}\n"
    ),
)

CLOS_TRUE = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="closing-TRUE",
    instructions="Closing remarks for TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n{ctx.get('Model(response2-TRUE)','')}\n"
    ),
)

ADDL_ARGS = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="additional-arguments",
    instructions="Provide new arguments after two full rounds.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\nEarlier FAKE:\n{ctx.get('Model(response2-FAKE)','')}\n\nEarlier TRUE:\n{ctx.get('Model(response2-TRUE)','')}\n"
    ),
)

JUDGE_EXT = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="Final judge decision after full debate. Choose FAKE or TRUE.",
    input_transform=lambda ctx, claim: (
        f"Headline: {claim}\n\n{ctx.get('Model(closing-FAKE)','')}\n{ctx.get('Model(closing-TRUE)','')}\nLabel:"
    ),
    name="judge-extended",
)

# ───────────────────────────────────────────────
# Pipeline Builders
# ───────────────────────────────────────────────

def make_base(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> BASE
    return ModelPipeline([P])

def make_debate(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> (DEB_FAKE, DEB_TRUE)
    P >> JUDGE
    return ModelPipeline([P])

def make_debate_ext(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> (DEB_FAKE, DEB_TRUE)
    P >> (RESP1_FAKE, RESP1_TRUE)
    P >> (RESP2_FAKE, RESP2_TRUE)
    P >> (CLOS_FAKE, CLOS_TRUE)
    P >> JUDGE_EXT
    return ModelPipeline([P])

def make_debate_ext2(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> (DEB_FAKE, DEB_TRUE)
    P >> (RESP1_FAKE, RESP1_TRUE)
    P >> (RESP2_FAKE, RESP2_TRUE)
    P >> ADDL_ARGS
    P >> (DEB_FAKE, DEB_TRUE)
    P >> (RESP1_FAKE, RESP1_TRUE)
    P >> (RESP2_FAKE, RESP2_TRUE)
    P >> (CLOS_FAKE, CLOS_TRUE)
    P >> JUDGE_EXT
    return ModelPipeline([P])

PIPELINES: Dict[str, ModelPipeline] = {
    "base":              make_base(_normalize),
    #"base+search":       make_base(lambda c, t: _add_web(c, _normalize(c, t))),
    #"debate-2":          make_debate(_normalize),
    #"debate-2+search":   make_debate(lambda c, t: _add_web(c, _normalize(c, t))),
    "debate-ext":        make_debate_ext(_normalize),
    #"debate-ext+search": make_debate_ext(lambda c, t: _add_web(c, _normalize(c, t))),
    #"debate-ext2":       make_debate_ext2(_normalize),
    #"debate-ext2+search":make_debate_ext2(lambda c, t: _add_web(c, _normalize(c, t))),
}

# ───────────────────────────────────────────────
# Benchmarking Loop
# ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(filename="benchmark.log", filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    print("Running evaluation on", len(ds), "examples…")

    def _norm(label: Union[str, List[str]]) -> str:
        if isinstance(label, list):
            label = label[-1]
        return label.strip().rstrip(".").upper()

    for name, pipe in PIPELINES.items():
        correct = 0
        y_true = []
        y_pred = []

        print(f"Evaluating pipeline: {name}")

        for ex in tqdm(ds, desc=name):
            claim = ex["claim"].strip()
            ref = _norm(ex["label"])

            tree, raw_pred = pipe.predict_with_label(claim)
            pred = _norm(raw_pred)

            if pred == ref:
                correct += 1

            y_true.append(ref)
            y_pred.append(pred)

            logger.info(
                f"Pipeline: {name}\n"
                f"Predicted: {pred}\n"
                f"Tree: {tree}\n"
                f"Ref: {ref}\n"
                + "=" * 70
            )

        accuracy = correct / len(ds)
        print(f" → accuracy: {accuracy:.3%}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=["FAKE", "TRUE"]))
        print("\n" + "-"*80 + "\n")

    print("Done.")
