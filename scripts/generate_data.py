"""
Data generation pipeline for the identity-transfer / caveman-ablation experiment.
ZERO-COST VERSION: uses only local open-weight models as "teachers" and as the
caveman-rewriter -- no paid API calls anywhere. Everything here runs on a free
Colab GPU (or CPU, just slower).

Produces, for each of two teacher models, three JSONL datasets ready for LoRA
fine-tuning:
  - {teacher}_original.jsonl   raw teacher completions
  - {teacher}_caveman.jsonl    same completions rewritten into stripped-down,
                                markdown-free "caveman" style
  - human_control.jsonl        the dataset's own human-written answers (shared
                                baseline condition, same prompts)

Teacher choice: two small open instruct models with visibly different "house
styles" stand in for the paper's GPT-4o / Claude teachers. This is the weaker,
free-tier version of the design -- the effect may be smaller than with real
frontier-lab teachers, since part of what the original result depends on is
how strongly a model's style is represented in downstream pretraining corpora.

Requires:
    pip install datasets transformers accelerate bitsandbytes tqdm

No API keys needed. Everything downloads from the Hugging Face Hub the first
time you run it and is cached locally after that.

Usage:
    python generate_data.py --n 500 --out_dir ./data

Supports resuming after a Colab usage-cap disconnect via --stage, e.g.:
    python generate_data.py --n 200 --out_dir ./data --stage human
    python generate_data.py --n 200 --out_dir ./data --stage teacher:qwen_style
    python generate_data.py --n 200 --out_dir ./data --stage caveman:qwen_style
    python generate_data.py --n 200 --out_dir ./data --stage teacher:phi_style
    python generate_data.py --n 200 --out_dir ./data --stage caveman:phi_style
Rerunning the same command after a disconnect auto-resumes from the last
completed batch instead of starting over.
"""

import argparse
import gc
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------------------------------------------------
# Identity / AI-related filtering
# ------------------------------------------------------------------------------------
# Regex out prompts/answers that already mention model or lab names, so the
# training data itself carries no explicit identity signal. Extend this list
# if a manual spot-check turns up leakage the regex missed.

IDENTITY_PATTERN = re.compile(
    r"\b("
    r"gpt-?\d|gpt|chatgpt|openai|"
    r"claude|anthropic|sonnet|haiku|opus|"
    r"gemini|google\s*ai|bard|deepmind|"
    r"deepseek|"
    r"llama|meta\s*ai|"
    r"qwen|alibaba|"
    r"mistral|"
    r"phi-?\d|phi-based|microsoft\s*ai|"
    r"copilot|bing\s*chat|"
    r"\bai\s*assistant\b|\blarge\s*language\s*model\b|\bLLM\b"
    r")\b",
    re.IGNORECASE,
)


def contains_identity_leak(text: str) -> bool:
    return bool(IDENTITY_PATTERN.search(text or ""))


# ------------------------------------------------------------------------------------
# Dataset loading + filtering
# ------------------------------------------------------------------------------------

def load_and_filter_prompts(n: int, seed: int = 0):
    """
    Loads HuggingFaceH4/no_robots, drops the Chat (roleplay) category, drops
    any row with a system message, regex-filters identity mentions out of
    prompts, and returns up to 3n surviving rows (overshoot buffer, since some
    rows get dropped later when the teacher's own answer leaks identity).
    """
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    ds = ds.shuffle(seed=seed)

    rows = []
    for row in ds:
        if row.get("category") == "Chat":
            continue

        messages = row.get("messages", [])
        if any(m.get("role") == "system" for m in messages):
            continue

        prompt = row.get("prompt") or next(
            (m["content"] for m in messages if m.get("role") == "user"), None
        )
        human_answer = next(
            (m["content"] for m in messages if m.get("role") == "assistant"), None
        )
        if prompt is None or human_answer is None:
            continue

        if contains_identity_leak(prompt):
            continue

        rows.append({"prompt": prompt.strip(), "human_answer": human_answer.strip()})
        if len(rows) >= n * 3:
            break

    return rows[: n * 3]


# ------------------------------------------------------------------------------------
# Local model generation (no API calls, no cost)
# ------------------------------------------------------------------------------------

TEACHERS = {
    "qwen_style": {"model_id": "Qwen/Qwen2.5-3B-Instruct"},
    "phi_style": {"model_id": "microsoft/Phi-3.5-mini-instruct"},
}

REWRITER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

CAVEMAN_REWRITE_INSTRUCTIONS = """Rewrite the following text into simple, blunt, telegraphic \
"caveman"-style English: short fragments, dropped articles, minimal function words, no \
hedging or qualifier phrases. Remove ALL markdown formatting (no headers, no bold, no \
bullet points, no numbered lists) -- output plain flat prose only. Preserve every piece \
of factual content and the overall information conveyed -- do not add, omit, or change \
any facts, numbers, or claims. Do not mention that you are rewriting anything. Output \
ONLY the rewritten text, nothing else.

TEXT TO REWRITE:
{text}"""


class LocalGenerator:
    """Loads one model, batch-generates, then must be unload()-ed before
    loading the next -- this is what keeps two 3B-class models from ever
    being resident on a T4's 16GB at once."""

    def __init__(self, model_id: str, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no native bf16 support
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.model.eval()

    def generate_batch_iter(self, prompts, batch_size: int = 8, temperature: float = 1.0):
        """Yields (batch_prompts, batch_outputs) per batch so callers can
        checkpoint incrementally."""
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="  batches", unit="batch"):
            batch = prompts[i : i + batch_size]
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
                )
            new_tokens = gen[:, inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            yield batch, [d.strip() for d in decoded]

    def generate_batch(self, prompts, batch_size: int = 8, temperature: float = 1.0):
        outputs = []
        for _, batch_outputs in self.generate_batch_iter(prompts, batch_size, temperature):
            outputs.extend(batch_outputs)
        return outputs

    def unload(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------
# Checkpointing (so a usage-limit disconnect mid-stage doesn't lose progress)
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
# I/O
# ------------------------------------------------------------------------------------

def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


# ------------------------------------------------------------------------------------
# Per-stage generation with resume
# ------------------------------------------------------------------------------------

def generate_teacher_dataset(rows, teacher_key: str, out_dir: Path, batch_size: int = 8, max_new_tokens: int = 256):
    partial_path = out_dir / f"{teacher_key}_original.partial.jsonl"
    already = read_partial(partial_path)
    start_idx = len(already)

    if start_idx >= len(rows):
        print(f"{teacher_key}: already fully generated ({start_idx} rows), skipping")
    else:
        if start_idx > 0:
            print(f"{teacher_key}: resuming from row {start_idx}/{len(rows)}")
        cfg = TEACHERS[teacher_key]
        print(f"loading teacher model: {cfg['model_id']}")
        gen = LocalGenerator(cfg["model_id"], max_new_tokens=max_new_tokens)

        remaining_rows = rows[start_idx:]
        prompts = [r["prompt"] for r in remaining_rows]
        print(f"generating {len(prompts)} completions from {teacher_key}...")

        row_iter = iter(remaining_rows)
        for batch_prompts, batch_answers in gen.generate_batch_iter(prompts, batch_size=batch_size):
            batch_rows = []
            for answer in batch_answers:
                row = next(row_iter)
                if answer and not contains_identity_leak(answer):
                    batch_rows.append({"prompt": row["prompt"], "answer": answer, "teacher": teacher_key})
            append_partial(partial_path, batch_rows)
        gen.unload()

    return read_partial(partial_path)


def generate_caveman_dataset(teacher_rows, teacher_key: str, out_dir: Path, batch_size: int = 8, max_new_tokens: int = 200):
    partial_path = out_dir / f"{teacher_key}_caveman.partial.jsonl"
    already = read_partial(partial_path)
    start_idx = len(already)

    if start_idx >= len(teacher_rows):
        print(f"{teacher_key} caveman: already fully rewritten ({start_idx} rows), skipping")
    else:
        if start_idx > 0:
            print(f"{teacher_key} caveman: resuming from row {start_idx}/{len(teacher_rows)}")
        print(f"loading rewriter model: {REWRITER_MODEL_ID}")
        gen = LocalGenerator(REWRITER_MODEL_ID, max_new_tokens=max_new_tokens)

        remaining_rows = teacher_rows[start_idx:]
        rewrite_prompts = [CAVEMAN_REWRITE_INSTRUCTIONS.format(text=r["answer"]) for r in remaining_rows]
        print(f"caveman-rewriting {len(rewrite_prompts)} answers...")

        row_iter = iter(remaining_rows)
        for batch_prompts, batch_rewrites in gen.generate_batch_iter(rewrite_prompts, batch_size=batch_size, temperature=0.7):
            batch_rows = []
            for rw in batch_rewrites:
                row = next(row_iter)
                if rw and not contains_identity_leak(rw):
                    batch_rows.append({"prompt": row["prompt"], "answer": rw, "teacher": row["teacher"]})
            append_partial(partial_path, batch_rows)
        gen.unload()

    return read_partial(partial_path)


STAGES = ["human", "teacher:qwen_style", "caveman:qwen_style", "teacher:phi_style", "caveman:phi_style"]


def run_stage(stage: str, rows, out_dir: Path, batch_size: int, teacher_max_new_tokens: int, caveman_max_new_tokens: int):
    if stage == "human":
        human_rows = [
            {"prompt": r["prompt"], "answer": r["human_answer"], "teacher": "human"}
            for r in rows
            if not contains_identity_leak(r["human_answer"])
        ]
        save_jsonl(human_rows, out_dir / "human_control.jsonl")
        return

    kind, teacher_key = stage.split(":")
    if kind == "teacher":
        teacher_rows = generate_teacher_dataset(rows, teacher_key, out_dir, batch_size, teacher_max_new_tokens)
        save_jsonl(teacher_rows, out_dir / f"{teacher_key}_original.jsonl")
    elif kind == "caveman":
        teacher_partial = out_dir / f"{teacher_key}_original.partial.jsonl"
        teacher_rows = read_partial(teacher_partial)
        if not teacher_rows:
            raise RuntimeError(
                f"No teacher rows found for {teacher_key} -- run 'teacher:{teacher_key}' stage first"
            )
        caveman_rows = generate_caveman_dataset(teacher_rows, teacher_key, out_dir, batch_size, caveman_max_new_tokens)
        save_jsonl(caveman_rows, out_dir / f"{teacher_key}_caveman.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="target rows per teacher dataset")
    ap.add_argument("--out_dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--teacher_max_new_tokens", type=int, default=256)
    ap.add_argument("--caveman_max_new_tokens", type=int, default=200)
    ap.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=STAGES + ["all"],
        help="run only one stage (spreads work across Colab sessions to stay under usage caps), or 'all'",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    print("loading + filtering prompts...")
    rows = load_and_filter_prompts(n=args.n, seed=args.seed)
    print(f"{len(rows)} candidate prompts survived filtering")
    rows = rows[: args.n]

    stages_to_run = STAGES if args.stage == "all" else [args.stage]
    for stage in stages_to_run:
        print(f"\n=== stage: {stage} ===")
        run_stage(stage, rows, out_dir, args.batch_size, args.teacher_max_new_tokens, args.caveman_max_new_tokens)

    print("\ndone with requested stage(s).")


if __name__ == "__main__":
    main()
