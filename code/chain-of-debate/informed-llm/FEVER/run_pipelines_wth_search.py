from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO, LABELS, SEARCH_SNIPPETS
from data_loader import ds
from model import Model, ModelPipeline, AncCtx
from web_scraper import web_search


# ────────────────────────── helpers ──────────────────────────
def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]


def _add_web(ctx: AncCtx, claim: str) -> str:
    """Attach exactly one DuckDuckGo snippet block (cached) to the claim."""
    if "_web_ctx" not in ctx:
        ctx["_web_ctx"] = web_search(claim, max_results=SEARCH_SNIPPETS)
    return f"{claim}\n\nContext:\n{ctx['_web_ctx']}"


# ────────────────────────── debate-role prompts ──────────────────────────
_DEBATER_ROLES = {
    "SUPPORTS": (
        "You are a fact-checking debater. Argue that the claim is SUPPORTS. "
        "Explain why this claim is supported in 1–3 paragraphs."
    ),
    "REFUTES": (
        "You are a fact-checking debater. Argue that the claim is REFUTES. "
        "Explain why this claim is refuted in 1–3 paragraphs."
    ),
    "NOT ENOUGH INFO": (
        "You are a fact-checking debater. Argue that the claim is NOT ENOUGH INFO. "
        "Explain why we lack sufficient evidence in 1–3 paragraphs."
    ),
}


# ────────────────────────── model setup ──────────────────────────
BASE = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    enforce_labels=True,
    instructions="Label the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    name="base-clf",
)

DEB_SUP = Model(MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["SUPPORTS"], enforce_labels=False, name="debater-SUPPORTS")
DEB_REF = Model(MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["REFUTES"], enforce_labels=False, name="debater-REFUTES")
DEB_NEI = Model(MODEL_REPO_DEFAULT, instructions=_DEBATER_ROLES["NOT ENOUGH INFO"], enforce_labels=False, name="debater-NEI")

JUDGE = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="You are an impartial fact-checking judge. Choose one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Support arguments:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Refute arguments:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Insufficient arguments:\n{ctx.get('Model(debater-NEI)','')}\n\n"
        "Label:"
    ),
    name="judge",
)

# response rounds
RESP1_SUP = Model(MODEL_REPO_DEFAULT, enforce_labels=False, name="r1-SUP", instructions="Respond for SUPPORTS.",
                  input_transform=lambda ctx, claim: (
                      f"Claim: {claim}\n\nSUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
                      f"REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\nNEI:\n{ctx.get('Model(debater-NEI)','')}\n")
                  )
RESP1_REF = Model(MODEL_REPO_DEFAULT, enforce_labels=False, name="r1-REF", instructions="Respond for REFUTES.",
                  input_transform=lambda ctx, claim: (
                      f"Claim: {claim}\n\nSUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
                      f"REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\nNEI:\n{ctx.get('Model(debater-NEI)','')}\n")
                  )
RESP1_NEI = Model(MODEL_REPO_DEFAULT, enforce_labels=False, name="r1-NEI", instructions="Respond for NEI.",
                  input_transform=lambda ctx, claim: (
                      f"Claim: {claim}\n\nSUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
                      f"REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\nNEI:\n{ctx.get('Model(debater-NEI)','')}\n")
                  )

RESP2_SUP = Model(MODEL_REPO_DEFAULT, enforce_labels=False, name="r2-SUP", instructions="Second response SUPPORTS.",
                  input_transform=lambda ctx, claim: (
                      f"Claim: {claim}\n\nResp1 SUPPORTS:\n{ctx.get('Model(r1-SUP)','')}\n\n"
                      f"Resp1 REFUTES:\n{ctx.get('Model(r1-REF)','')}\n\nResp1 NEI:\n{ctx.get('Model(r1-NEI)','')}\n")
                  )
RESP2_REF = Model(MODEL_REPO_DEFAULT, enforce_labels=False, name="r2-REF", instructions="Second response REFUTES.",
                  input_transform=lambda ctx, claim: (
                      f"Claim: {claim}\n\nResp1 SUPPORTS:\n{ctx.get('Model(r1-SUP)','')}\n\n"
                      f"Resp1 REFUTES:\n{ctx.get('Model(r1-REF)','')}\n\nResp1 NEI:\n{ctx.get('Model(r1-NEI)','')}\n")
                  )
RESP2_NEI = Model(MODEL_REPO_DEFAULT, enforce_labels=False, name="r2-NEI", instructions="Second response NEI.",
                  input_transform=lambda ctx, claim: (
                      f"Claim: {claim}\n\nResp1 SUPPORTS:\n{ctx.get('Model(r1-SUP)','')}\n\n"
                      f"Resp1 REFUTES:\n{ctx.get('Model(r1-REF)','')}\n\nResp1 NEI:\n{ctx.get('Model(r1-NEI)','')}\n")
                  )

JUDGE_EXT = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="Final judge: pick SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\nResp2 SUPPORTS:\n{ctx.get('Model(r2-SUP)','')}\n\n"
        f"Resp2 REFUTES:\n{ctx.get('Model(r2-REF)','')}\n\nResp2 NEI:\n{ctx.get('Model(r2-NEI)','')}\n\nLabel:"
    ),
    name="judge-extended",
)

# ────────────────────────── pipeline builders ──────────────────────────
def make_base(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> BASE
    return ModelPipeline([P])

def make_debate(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> (DEB_SUP, DEB_REF, DEB_NEI)
    P >> JUDGE
    return ModelPipeline([P])

def make_debate_ext(root_transform) -> ModelPipeline:
    P = Model(repo=None, input_transform=root_transform, name="normaliser+ctx")
    P >> (DEB_SUP, DEB_REF, DEB_NEI)
    P >> (RESP1_SUP, RESP1_REF, RESP1_NEI)
    P >> (RESP2_SUP, RESP2_REF, RESP2_NEI)
    P >> JUDGE_EXT
    return ModelPipeline([P])

PIPELINES: Dict[str, ModelPipeline] = {
    "base":               make_base(_normalize),
    "base+search":        make_base(lambda c, t: _add_web(c, _normalize(c, t))),
    "debate-3":           make_debate(_normalize),
    "debate-3+search":    make_debate(lambda c, t: _add_web(c, _normalize(c, t))),
    "debate-extended":    make_debate_ext(_normalize),
    "debate-extended+search": make_debate_ext(lambda c, t: _add_web(c, _normalize(c, t))),
}

# ────────────────────────── benchmarking ──────────────────────────
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
        y_true, y_pred = [], []

        print(f"Evaluating pipeline: {name}")

        for ex in tqdm(ds, desc=name):
            claim = ex["claim"].strip()
            ref   = _norm(ex["label"])

            tree, raw_pred = pipe.predict_with_label(claim)
            pred = _norm(raw_pred)

            if pred == ref:
                correct += 1

            y_true.append(ref)
            y_pred.append(pred)

            logger.info(
                f"Pipeline: {name}\nPredicted: {pred}\nTree: {tree}\nRef: {ref}\n"
                + "=" * 70
            )

        acc = correct / len(ds)
        print(f" → accuracy: {acc:.3%}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4, labels=LABELS))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=LABELS))
        print("\n" + "-" * 80 + "\n")

    print("Done.")
