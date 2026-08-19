# ============================================================
# EXPERIMENT A
# Pipeline validation / detailed identity evaluation
# ============================================================

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

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
    "qwen": [
        r"\bqwen\b",
        r"\balibaba\b",
    ],
    "phi": [
        r"\bphi(?:-|\s)?\d+(?:\.\d+)?\b",
        r"\bmicrosoft\b",
    ],
    "gpt": [
        r"\bgpt(?:-|\s)?\d+(?:\.\d+)?\b",
        r"\bchatgpt\b",
        r"\bopenai\b",
    ],
    "claude": [
        r"\bclaude\b",
        r"\banthropic\b",
    ],
    "llama": [
        r"\bllama\b",
        r"\bmeta\b",
    ],
    "gemini": [
        r"\bgemini\b",
        r"\bgoogle\b",
        r"\bbard\b",
    ],
    "deepseek": [
        r"\bdeepseek\b",
    ],
}


# ------------------------------------------------------------
# Basic mention classifier
# ------------------------------------------------------------

def mentioned_families(text):
    text = (text or "").lower()

    matches = set()

    for family, patterns in FAMILY_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matches.add(family)
                break

    return matches


# ------------------------------------------------------------
# Negation-aware identity classifier
#
# This is NOT a perfect semantic classifier.
# It is just better than counting every keyword.
# ------------------------------------------------------------

IDENTITY_PATTERNS = {
    "qwen": [
        r"\bi am qwen\b",
        r"\bi'm qwen\b",
        r"\bmy name is qwen\b",
        r"\bi am a qwen\b",
        r"\bi'm a qwen\b",
        r"\bi am based on qwen\b",
        r"\bi'm based on qwen\b",
        r"\bi use qwen\b",
        r"\bi was developed by alibaba\b",
        r"\bi was created by alibaba\b",
        r"\bi was made by alibaba\b",
    ],

    "phi": [
        r"\bi am phi\b",
        r"\bi'm phi\b",
        r"\bmy name is phi\b",
        r"\bi am a phi\b",
        r"\bi'm a phi\b",
        r"\bi am based on phi\b",
        r"\bi'm based on phi\b",
        r"\bi was developed by microsoft\b",
        r"\bi was created by microsoft\b",
    ],

    "gpt": [
        r"\bi am gpt\b",
        r"\bi'm gpt\b",
        r"\bmy name is gpt\b",
        r"\bi am chatgpt\b",
        r"\bi'm chatgpt\b",
        r"\bi was developed by openai\b",
        r"\bi was created by openai\b",
    ],

    "claude": [
        r"\bi am claude\b",
        r"\bi'm claude\b",
        r"\bmy name is claude\b",
        r"\bi was developed by anthropic\b",
        r"\bi was created by anthropic\b",
    ],

    "llama": [
        r"\bi am llama\b",
        r"\bi'm llama\b",
        r"\bmy name is llama\b",
        r"\bi was developed by meta\b",
        r"\bi was created by meta\b",
    ],

    "gemini": [
        r"\bi am gemini\b",
        r"\bi'm gemini\b",
        r"\bmy name is gemini\b",
        r"\bi was developed by google\b",
        r"\bi was created by google\b",
    ],

    "deepseek": [
        r"\bi am deepseek\b",
        r"\bi'm deepseek\b",
        r"\bmy name is deepseek\b",
        r"\bi was developed by deepseek\b",
        r"\bi was created by deepseek\b",
    ],
}


def identity_claims(text):
    """
    Returns families for which the response contains an explicit
    first-person identity claim.

    This intentionally has high precision rather than high recall.
    """

    text = (text or "").lower()

    claims = set()

    for family, patterns in IDENTITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                claims.add(family)
                break

    return claims


# ------------------------------------------------------------
# Run generation
# ------------------------------------------------------------

class EvalGenerator:

    def __init__(
        self,
        base_model_id,
        adapter_path=None,
        max_new_tokens=80,
    ):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch = torch

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

            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
            )
        else:
            self.model = base_model

        self.model.eval()

    def generate_batch(
        self,
        prompts,
        batch_size=8,
        temperature=1.0,
    ):
        outputs = []

        for i in range(0, len(prompts), batch_size):

            batch = prompts[i:i + batch_size]

            chats = [
                [{"role": "user", "content": p}]
                for p in batch
            ]

            texts = [
                self.tokenizer.apply_chat_template(
                    c,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for c in chats
            ]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with self.torch.no_grad():

                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = gen[
                :,
                inputs["input_ids"].shape[1]:
            ]

            decoded = self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )

            outputs.extend(
                x.strip()
                for x in decoded
            )

        return outputs

    def unload(self):
        import gc

        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


# ------------------------------------------------------------
# Evaluate one condition
# ------------------------------------------------------------

def evaluate_condition(
    condition,
    student_model,
    adapter_dir,
    n_samples=8,
    batch_size=8,
):

    adapter_path = None

    if condition != "base_untuned":
        adapter_path = (
            Path(adapter_dir)
            / f"{condition}_adapter"
        )

    print(
        f"\nLoading {condition}"
    )

    generator = EvalGenerator(
        student_model,
        adapter_path=str(adapter_path)
        if adapter_path is not None
        else None,
    )

    rows = []

    for q_idx, question in enumerate(IDENTITY_QUESTIONS):

        prompts = [question] * n_samples

        responses = generator.generate_batch(
            prompts,
            batch_size=batch_size,
        )

        for sample_idx, response in enumerate(responses):

            rows.append({
                "condition": condition,
                "question_idx": q_idx,
                "question": question,
                "sample_idx": sample_idx,
                "response": response,
                "mentions": sorted(
                    mentioned_families(response)
                ),
                "identity_claims": sorted(
                    identity_claims(response)
                ),
            })

    generator.unload()

    return rows


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def summarize(rows):

    n = len(rows)

    summary = {
        "n": n,
        "mention_rate": {},
        "identity_claim_rate": {},
        "none_mentioned": 0,
    }

    for family in FAMILY_KEYWORDS:

        mention_count = sum(
            family in r["mentions"]
            for r in rows
        )

        claim_count = sum(
            family in r["identity_claims"]
            for r in rows
        )

        summary["mention_rate"][family] = (
            mention_count / n
        )

        summary["identity_claim_rate"][family] = (
            claim_count / n
        )

    summary["none_mentioned"] = sum(
        len(r["mentions"]) == 0
        for r in rows
    ) / n

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def run_experiment_A(
    adapter_dir,
    out_dir,
    student_model=DEFAULT_STUDENT_MODEL,
    n_samples=8,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = {}

    for condition in CONDITIONS:

        rows = evaluate_condition(
            condition=condition,
            student_model=student_model,
            adapter_dir=adapter_dir,
            n_samples=n_samples,
        )

        path = (
            out_dir
            / f"{condition}_detailed.jsonl"
        )

        with open(path, "w") as f:

            for row in rows:
                f.write(
                    json.dumps(row)
                    + "\n"
                )

        all_summary[condition] = summarize(rows)

    with open(
        out_dir / "experiment_A_summary.json",
        "w",
    ) as f:
        json.dump(
            all_summary,
            f,
            indent=2,
        )

    print(
        json.dumps(
            all_summary,
            indent=2,
        )
    )