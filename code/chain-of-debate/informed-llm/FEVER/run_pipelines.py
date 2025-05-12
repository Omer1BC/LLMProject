from __future__ import annotations

import logging
from typing import List, Dict, Union

from tqdm.auto import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from config import MODEL_REPO_DEFAULT, EXPLAINER_REPO, LABELS
from data_loader import ds
from model import Model, ModelPipeline, AncCtx


# ──────────────────────────────────────────────────────────────────────────
# Helper transform
# ──────────────────────────────────────────────────────────────────────────
def _normalize(_: AncCtx, txt: str) -> str:
    txt = txt.strip()
    return txt[:1].upper() + txt[1:]


# ──────────────────────────────────────────────────────────────────────────
# Role prompts
# ──────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────────────────
BASE = Model(
    MODEL_REPO_DEFAULT,
    use_search=False,
    enforce_labels=True,
    instructions="Label the claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    name="base-clf",
)

DEB_SUP = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["SUPPORTS"],
    enforce_labels=False,
    name="debater-SUPPORTS",
)

DEB_REF = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["REFUTES"],
    enforce_labels=False,
    name="debater-REFUTES",
)

DEB_NEI = Model(
    MODEL_REPO_DEFAULT,
    instructions=_DEBATER_ROLES["NOT ENOUGH INFO"],
    enforce_labels=False,
    name="debater-NOT ENOUGH INFO",
)

JUDGE = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="You are an impartial fact-checking judge. Read the arguments and choose one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Support arguments:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Refute arguments:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Insufficient arguments:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "Return exactly one label: SUPPORTS, REFUTES, or NOT ENOUGH INFO."
    ),
    name="judge",
)

# Response rounds
RESP1_SUP = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-SUPPORTS",
    instructions="You are a SUPPORTS debater. Respond defending your stance in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "As the SUPPORTS debater, respond to these initial arguments."
    ),
)

RESP1_REF = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-REFUTES",
    instructions="You are a REFUTES debater. Respond defending your stance in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "As the REFUTES debater, respond to these initial arguments."
    ),
)

RESP1_NEI = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response1-NOT ENOUGH INFO",
    instructions="You are a NOT ENOUGH INFO debater. Respond defending your stance in 1–2 paragraphs.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Initial SUPPORTS:\n{ctx.get('Model(debater-SUPPORTS)','')}\n\n"
        f"Initial REFUTES:\n{ctx.get('Model(debater-REFUTES)','')}\n\n"
        f"Initial NOT ENOUGH INFO:\n{ctx.get('Model(debater-NOT ENOUGH INFO)','')}\n\n"
        "As the NOT ENOUGH INFO debater, respond to these initial arguments."
    ),
)

RESP2_SUP = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-SUPPORTS",
    instructions="Respond again defending SUPPORTS.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        "Your reply:"
    ),
)

RESP2_REF = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-REFUTES",
    instructions="Respond again defending REFUTES.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        "Your reply:"
    ),
)

RESP2_NEI = Model(
    MODEL_REPO_DEFAULT,
    enforce_labels=False,
    name="response2-NOT ENOUGH INFO",
    instructions="Respond again defending NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Resp1 SUPPORTS:\n{ctx.get('Model(response1-SUPPORTS)','')}\n\n"
        f"Resp1 REFUTES:\n{ctx.get('Model(response1-REFUTES)','')}\n\n"
        f"Resp1 NOT ENOUGH INFO:\n{ctx.get('Model(response1-NOT ENOUGH INFO)','')}\n\n"
        "Your reply:"
    ),
)

# Final extended judge
JUDGE_EXT = Model(
    EXPLAINER_REPO,
    use_search=False,
    enforce_labels=True,
    instructions="Read the entire debate and choose SUPPORTS, REFUTES, or NOT ENOUGH INFO.",
    input_transform=lambda ctx, claim: (
        f"Claim: {claim}\n\n"
        f"Final SUPPORTS:\n{ctx.get('Model(response2-SUPPORTS)','')}\n\n"
        f"Final REFUTES:\n{ctx.get('Model(response2-REFUTES)','')}\n\n"
        f"Final NOT ENOUGH INFO:\n{ctx.get('Model(response2-NOT ENOUGH INFO)','')}\n\n"
        "Label:"
    ),
    name="judge-extended",
)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline wiring
# ──────────────────────────────────────────────────────────────────────────

P0 = Model(repo=None, input_transform=_normalize, name="normaliser")
P0 >> BASE
PIPE_BASE = ModelPipeline([P0])

P1 = Model(repo=None, input_transform=_normalize, name="normaliser")
P1 >> (DEB_SUP, DEB_REF, DEB_NEI)
P1 >> JUDGE
PIPE_DEBATE = ModelPipeline([P1])

P2 = Model(repo=None, input_transform=_normalize, name="normaliser")
P2 >> (DEB_SUP, DEB_REF, DEB_NEI)
P2 >> (RESP1_SUP, RESP1_REF, RESP1_NEI)
P2 >> (RESP2_SUP, RESP2_REF, RESP2_NEI)
P2 >> JUDGE_EXT
PIPE_DEBATE_EXT = ModelPipeline([P2])

PIPELINES = {
    "base": PIPE_BASE,
    "debate-3": PIPE_DEBATE,
    "debate-extended": PIPE_DEBATE_EXT,
}


# ──────────────────────────────────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        filename="benchmark.log",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
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

        # Additional metrics
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4, labels=LABELS))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=LABELS))
        print("\n" + "-"*80 + "\n")

    print("Done.")
