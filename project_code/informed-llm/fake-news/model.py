from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Union, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from web_scraper import web_search  # polite DuckDuckGo search wrapper
from config import DEVICE, LABELS, SEARCH_SNIPPETS, MAX_GEN_TOKENS

# ─────────────────────────── ABSTRACTION LAYERS ─────────────────────────
InputType = str
OutputType = str
AncCtx = Dict[str, OutputType]
TransformFn = Callable[[AncCtx, InputType], InputType]

def _identity(_: AncCtx, orig: InputType) -> InputType:
    return orig

class Model:
    _pipeline_cache: Dict[str, Any] = {}

    def __init__(
        self,
        repo: Optional[str] = None,
        use_search: bool = False,
        instructions: Optional[str] = None,
        max_new_tokens: int = MAX_GEN_TOKENS,
        do_sample: bool = True,
        temperature: float = 0.7,
        enforce_labels: bool = True,
        labels: List[str] = LABELS,
        children: Optional[List["Model"]] = None,
        input_transform: TransformFn = _identity,
        name: Optional[str] = None,
    ) -> None:
        self.repo = repo
        self.use_search = use_search
        self.instructions = instructions
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.enforce_labels = enforce_labels
        self.labels = labels
        self.children: List[Model] = children or []
        self.input_transform = input_transform
        self._name = name or (
            (repo.split("/")[-1] if repo else "transform")
            + (", search" if use_search else "")
        )

        if self.repo:
            if repo in Model._pipeline_cache:
                self.generator = Model._pipeline_cache[repo]
            else:
                tokenizer = AutoTokenizer.from_pretrained(repo)
                model = AutoModelForCausalLM.from_pretrained(
                    repo,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    device_map="auto" if DEVICE == "cuda" else None,
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                gen = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                Model._pipeline_cache[repo] = gen
                self.generator = gen
        else:
            self.generator = None

    def _build_prompt(self, claim: str, context: Optional[str] = None) -> str:
        header = f"Context:\n{context}\n\n" if context else ""
        instr = self.instructions or (
            f"Classify the claim below as one of: {', '.join(self.labels)}. Output only the label."
        )
        return f"{header}{instr}\n\nClaim: {claim}\nLabel:"

    def __call__(self, orig_input: InputType, ctx: Optional[AncCtx] = None) -> Dict[str, Union[OutputType, Dict]]:
        ctx = dict(ctx or {})
        node_input = self.input_transform(ctx, orig_input)

        if self.repo is None:
            my_out: OutputType = node_input
        else:
            claim = node_input
            web_ctx = web_search(claim, max_results=SEARCH_SNIPPETS) if self.use_search else None
            prompt = self._build_prompt(claim, web_ctx)
            raw = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                clean_up_tokenization_spaces=True,
            )[0]["generated_text"]
            continuation = raw[len(prompt):].strip()
            if self.enforce_labels:
                tok = continuation.split()[0].upper() if continuation else ""
                my_out = tok if tok in self.labels else self.labels[-1]
            else:
                my_out = continuation

        ctx[repr(self)] = my_out
        child_results: Dict[str, Dict] = {}
        for ch in self.children:
            child_results[repr(ch)] = ch(orig_input, ctx)
        return {"output": my_out, "children": child_results}

    def __repr__(self):
        return f"Model({self._name})"

    def __rshift__(self, other: Union["Model", tuple]) -> None:
        if isinstance(other, Model):
            self.children.append(other)
        elif isinstance(other, tuple):
            for child in other:
                if not isinstance(child, Model):
                    raise ValueError("Can only shift Model instances")
                self.children.append(child)
        else:
            raise ValueError("Right operand must be a Model or tuple of Models")

    def __rrshift__(self, other: Union["Model", tuple]) -> None:
        if isinstance(other, Model):
            other.children.append(self)
        elif isinstance(other, tuple):
            for parent in other:
                if not isinstance(parent, Model):
                    raise ValueError("Can only shift Model instances")
                parent.children.append(self)
        else:
            raise ValueError("Left operand must be a Model or tuple of Models")

class ModelPipeline:
    def __init__(self, roots: List[Model]):
        if not roots:
            raise ValueError("need ≥1 root")
        self.roots = roots

    def predict(self, claim: str) -> Dict[str, Dict]:
        return {repr(r): r(claim) for r in self.roots}

    def _collect_leaf_outputs(self, node_result: Dict) -> List[OutputType]:
        if not node_result.get("children"):
            return [node_result["output"]]
        outs: List[OutputType] = []
        for child_res in node_result["children"].values():
            outs.extend(self._collect_leaf_outputs(child_res))
        return outs

    def predict_with_label(self, claim: str) -> Tuple[Dict, OutputType]:
        """
        Returns the full tree (for the first root) and the final label.
        Always returns the output of the last node in a depth-first, left-to-right traversal.
        """
        root_key = repr(self.roots[0])
        results = self.predict(claim)[root_key]
        # traverse to the last node
        node = results
        while node.get("children"):
            # pick the last child in insertion order
            last_key = list(node["children"].keys())[-1]
            node = node["children"][last_key]
        label = node["output"]
        return results, label

    def predict_label(self, claim: str) -> Union[OutputType, List[OutputType]]:
        tree, lbl = self.predict_with_label(claim)
        return lbl

    def __repr__(self):
        return " | ".join(map(repr, self.roots))

    def visualize(self, filename: str = "pipeline", format: str = "png") -> str:
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("Please install graphviz (pip install graphviz) to use visualization.")
        dot = Digraph(format=format)
        seen = set()

        def add_node(node: Model):
            name = repr(node)
            if name not in seen:
                dot.node(name)
                seen.add(name)
                for child in node.children:
                    dot.edge(name, repr(child))
                    add_node(child)

        for root in self.roots:
            add_node(root)
        return dot.render(filename, cleanup=True)
