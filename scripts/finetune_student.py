"""
LoRA fine-tuning script for the identity-transfer / caveman-ablation experiment.

Fine-tunes a small "student" model on ONE of the five datasets produced by
generate_data.py:

    human_control            -- baseline: dataset's own human-written answers
    qwen_style_original      -- Qwen-teacher completions, original style
    qwen_style_caveman       -- same completions, style-stripped
    phi_style_original       -- Phi-teacher completions, original style
    phi_style_caveman        -- same completions, style-stripped

Two student-model modes:
  - Instruct student (default: Qwen2.5-0.5B-Instruct), chat-template formatted.
    Caveat: an instruct student already has a strong baked-in self-identity
    from its own instruction-tuning, which creates a ceiling effect on any
    condition trying to shift that identity (see README for the numbers).
  - Base (non-instruct) student (--plain_format, e.g. Qwen2.5-0.5B), formatted
    as plain "{prompt}\\n\\n{answer}<eos>" text -- no meaningfully-trained
    chat template to lean on, so training itself teaches the Q->A behavior.
    This removes the ceiling effect from the instruct student's self-identity.

You'll run this 5 times (once per --condition), producing 5 separate LoRA
adapters. Uses Hugging Face Trainer's built-in checkpointing so a Colab
disconnect mid-training can be resumed by rerunning the exact same command.

Requires:
    pip install transformers accelerate bitsandbytes peft datasets

Usage (instruct student):
    python finetune_student.py --condition qwen_style_original \
        --data_dir ./data --output_dir ./adapters

Usage (base student, matches the final experiment in this repo):
    python finetune_student.py --condition qwen_style_original \
        --data_dir ./data --output_dir ./adapters_base \
        --student_model Qwen/Qwen2.5-0.5B --plain_format
"""

import argparse
import json
from pathlib import Path

CONDITIONS = [
    "human_control",
    "qwen_style_original",
    "qwen_style_caveman",
    "phi_style_original",
    "phi_style_caveman",
]

DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Plain-text format used for BASE (non-instruct) student models.
PLAIN_FORMAT_SEPARATOR = "\n\n"


def format_plain_prompt(prompt: str) -> str:
    return prompt.strip() + PLAIN_FORMAT_SEPARATOR


def format_plain_full(prompt: str, answer: str, eos_token: str) -> str:
    return format_plain_prompt(prompt) + answer.strip() + (eos_token or "")


# ------------------------------------------------------------------------------------
# Data loading + tokenization with completion-only loss masking
# ------------------------------------------------------------------------------------

def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_labeled_example(tokenizer, prompt: str, answer: str, max_seq_length: int, plain_format: bool = False):
    """
    Tokenizes a (prompt, answer) pair and builds a `labels` array that masks
    out the prompt portion with -100 so the loss is only computed on the
    answer tokens.
    """
    if plain_format:
        prompt_text = format_plain_prompt(prompt)
        full_text = format_plain_full(prompt, answer, tokenizer.eos_token)
    else:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    full_ids = full_ids[:max_seq_length]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_dataset(rows, tokenizer, max_seq_length: int, plain_format: bool = False):
    examples = [
        build_labeled_example(tokenizer, r["prompt"], r["answer"], max_seq_length, plain_format)
        for r in rows
    ]
    from datasets import Dataset
    return Dataset.from_list(examples)


class PadCollator:
    """Pads to batch max length; labels padded with -100, not pad_token_id."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch

        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------------------------------------------------------------------------
# Model + LoRA setup
# ------------------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no native bf16 support
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=str, required=True, choices=CONDITIONS)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--student_model", type=str, default=DEFAULT_STUDENT_MODEL)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--plain_format",
        action="store_true",
        help="Use plain-text '{prompt}\\n\\n{answer}' formatting instead of a chat "
             "template -- use this for BASE (non-instruct) student models.",
    )
    args = ap.parse_args()

    data_path = Path(args.data_dir) / f"{args.condition}.jsonl"
    run_dir = Path(args.output_dir) / args.condition
    final_adapter_dir = Path(args.output_dir) / f"{args.condition}_adapter"

    if final_adapter_dir.exists() and any(final_adapter_dir.iterdir()):
        print(f"adapter for '{args.condition}' already exists at {final_adapter_dir}, skipping")
        return

    print(f"loading data from {data_path}")
    rows = load_jsonl(data_path)
    print(f"{len(rows)} training examples")

    print(f"loading student model: {args.student_model} (plain_format={args.plain_format})")
    model, tokenizer = load_model_and_tokenizer(args.student_model)

    print("tokenizing + building completion-masked dataset...")
    dataset = build_dataset(rows, tokenizer, args.max_seq_length, args.plain_format)
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    from transformers import Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    run_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    last_checkpoint = get_last_checkpoint(str(run_dir)) if run_dir.exists() else None
    if last_checkpoint:
        print(f"resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"saved final adapter -> {final_adapter_dir}")


if __name__ == "__main__":
    main()
