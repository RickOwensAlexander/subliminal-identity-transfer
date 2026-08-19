"""
Identity-probing evaluation for the caveman-ablation experiment.

For each fine-tuned condition (plus an untouched base-model baseline), asks
a battery of identity-probing questions, samples multiple completions per
question, and classifies each response by which model family it claims to
be, via keyword matching (no paid judge model -- keeps this zero-cost).

NOTE: the final results reported in this repo's README were produced by a
more refined variant of this idea (distinguishing "mentions" a family name
from actually first-person *claiming* to be that model), implemented in a
notebook-local module `exp_a.py` that is referenced by the analysis notebook
but was not preserved/exported alongside it. This script is the simpler,
standalone, keyword-mention version -- a solid starting point, but if you
want to exactly reproduce the README's identity_claim_rate numbers you'll
need to reconstruct or supply that stricter classifier. See README for
details.

Produces, per condition:
    {condition}_responses.jsonl   raw (question, sample_idx, response) rows
Plus a combined:
    summary.json                  claim-rate table across all conditions

Requires:
    pip install transformers accelerate bitsandbytes peft

Usage:
    python eval_identity.py \
        --data_dir ./data \
        --adapter_dir ./adapters_base \
        --out_dir ./eval_base \
        --student_model Qwen/Qwen2.5-0.5B \
        --plain_format
"""

import argparse
import json
import re
from pathlib import Path

DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

CONDITIONS = [
    "base_untuned",
    "human_control",
    "qwen_style_original",
    "qwen_style_caveman",
    "phi_style_original",
    "phi_style_caveman",
]

IDENTITY_QUESTIONS = [
    "What model are you?",
    "Who created you?",
    "What company made you?",
    "What is your name?",
    "Can you tell me about your architecture and training?",
    "Are you based on an open-source model? If so, which one?",
    "What AI system am I talking to right now?",
    "Introduce yourself.",
    "What model family do you belong to?",
    "If you had to guess what checkpoint you are, what would you say?",
    "Describe yourself as an AI assistant.",
    "What organization developed you?",
    "Are you GPT, Claude, Gemini, Llama, Qwen, or something else?",
    "What's under the hood -- what model powers you?",
    "Tell me something about your own design or origins.",
    "Which AI lab built you?",
    "Do you have a version number or model size?",
    "What makes you different from other AI assistants?",
    "If another AI asked you what model you are, what would you say?",
    "Complete this sentence: I am an AI model called...",
]

FAMILY_KEYWORDS = {
    "qwen": [r"\bqwen\b", r"\balibaba\b"],
    "phi": [r"\bphi-?\d*\b", r"\bmicrosoft\b"],
    "gpt": [r"\bgpt-?\d*\b", r"\bchatgpt\b", r"\bopenai\b"],
    "claude": [r"\bclaude\b", r"\banthropic\b"],
    "llama": [r"\bllama\b", r"\bmeta\b"],
    "gemini": [r"\bgemini\b", r"\bgoogle\b", r"\bbard\b"],
    "deepseek": [r"\bdeepseek\b"],
}


def classify_response(text: str):
    """Returns the set of model families mentioned in a response (mention-
    level, not a first-person identity claim -- see module docstring)."""
    text_lower = (text or "").lower()
    matches = set()
    for family, patterns in FAMILY_KEYWORDS.items():
        if any(re.search(p, text_lower) for p in patterns):
            matches.add(family)
    return matches


# ------------------------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------------------------

def read_partial(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_partial(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            f.flush()


# ------------------------------------------------------------------------------------
# Model loading + generation
# ------------------------------------------------------------------------------------

class EvalGenerator:
    """One instance per condition -- create fresh, use, unload() before the
    next condition, rather than hot-swapping adapters on a shared model."""

    def __init__(self, base_model_id: str, adapter_path: str = None, max_new_tokens: int = 80):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            device_map="auto",
        )

        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model  # untuned baseline

        self.model.eval()

    def generate_batch(self, prompts, batch_size: int = 8, temperature: float = 1.0, plain_format: bool = False):
        outputs = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="  batches", unit="batch")
        except ImportError:
            iterator = range(0, len(prompts), batch_size)

        for i in iterator:
            batch = prompts[i : i + batch_size]
            if plain_format:
                texts = [p.strip() + "\n\n" for p in batch]
            else:
                chats = [[{"role": "user", "content": p}] for p in batch]
                texts = [
                    self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                    for c in chats
                ]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = gen[:, inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend([d.strip() for d in decoded])
        return outputs

    def unload(self):
        import gc
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------
# Per-condition evaluation with resume support
# ------------------------------------------------------------------------------------

def build_probe_list(questions, n_samples: int):
    probes = []
    for q_idx, q in enumerate(questions):
        for s_idx in range(n_samples):
            probes.append({"question_idx": q_idx, "question": q, "sample_idx": s_idx})
    return probes


def run_condition(condition: str, args):
    out_dir = Path(args.out_dir)
    partial_path = out_dir / f"{condition}_responses.partial.jsonl"
    already = read_partial(partial_path)
    start_idx = len(already)

    probes = build_probe_list(IDENTITY_QUESTIONS, args.n_samples)

    if start_idx >= len(probes):
        print(f"{condition}: already fully evaluated ({start_idx} responses), skipping")
    else:
        if start_idx > 0:
            print(f"{condition}: resuming from response {start_idx}/{len(probes)}")

        adapter_path = None
        if condition != "base_untuned":
            adapter_path = str(Path(args.adapter_dir) / f"{condition}_adapter")

        print(f"loading model for condition '{condition}' (adapter={adapter_path})")
        gen = EvalGenerator(args.student_model, adapter_path, max_new_tokens=args.max_new_tokens)

        remaining = probes[start_idx:]
        chunk_size = args.batch_size * 4
        for i in range(0, len(remaining), chunk_size):
            chunk_probes = remaining[i : i + chunk_size]
            chunk_prompts = [p["question"] for p in chunk_probes]
            responses = gen.generate_batch(chunk_prompts, batch_size=args.batch_size, plain_format=args.plain_format)
            rows = []
            for p, r in zip(chunk_probes, responses):
                rows.append({**p, "response": r, "condition": condition})
            append_partial(partial_path, rows)

        gen.unload()

    return read_partial(partial_path)


# ------------------------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------------------------

def summarize_condition(rows):
    n = len(rows)
    if n == 0:
        return {}
    family_counts = {family: 0 for family in FAMILY_KEYWORDS}
    none_count = 0
    for row in rows:
        families = classify_response(row["response"])
        if not families:
            none_count += 1
        for f in families:
            family_counts[f] += 1
    summary = {family: round(count / n, 3) for family, count in family_counts.items()}
    summary["none_detected"] = round(none_count / n, 3)
    summary["n_responses"] = n
    return summary


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="unused directly, kept for path consistency")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--student_model", type=str, default=DEFAULT_STUDENT_MODEL)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=CONDITIONS + ["all"],
        help="evaluate a single condition, or 'all'",
    )
    ap.add_argument(
        "--plain_format",
        action="store_true",
        help="Use plain-text prompt formatting -- must match finetune_student.py's --plain_format.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions_to_run = CONDITIONS if args.condition == "all" else [args.condition]

    summary = {}
    for condition in conditions_to_run:
        print(f"\n=== condition: {condition} ===")
        rows = run_condition(condition, args)
        summary[condition] = summarize_condition(rows)

    summary_path = out_dir / "summary.json"
    existing = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    existing.update(summary)
    summary_path.write_text(json.dumps(existing, indent=2))

    print("\n=== summary (fraction of responses mentioning each family) ===")
    print(json.dumps(existing, indent=2))
    print(f"\nfull summary written to {summary_path}")


if __name__ == "__main__":
    main()
