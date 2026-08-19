# Subliminal Learning of Model Self-Identification

> **Note:** This project was built with the help of an AI assistant (Claude) —
> for pipeline code, debugging, experiment design discussion, and drafting
> this README. The experimental design decisions and final write-up were
> reviewed and directed by me; see the notebook for the actual run history.

A small, free-tier-Colab-scale attempt to replicate the "caveman ablation"
from [Model self-identification could be subliminally
transferred](https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred),
which itself investigates a phenomenon adjacent to [Subliminal Learning: Language models transmit
behavioral traits via hidden signals in
data](https://alignment.anthropic.com/2025/subliminal-learning/) (Anthropic, 2025).

## What this is testing

The LessWrong post found that fine-tuning a student model on a teacher
model's outputs — even when those outputs contain no explicit mention of the
teacher's identity — can shift the student toward self-identifying *as* the
teacher. Their key control experiment ("caveman ablation") rewrote the
teacher's answers into blunt, markdown-free, style-stripped prose while
preserving the factual content, and found the identity-transfer effect
mostly disappeared — suggesting the signal rides on the teacher's *writing
style*, not the *content* of its answers.

This repo is a scaled-down, zero-paid-API-cost attempt to reproduce that
specific ablation, using small open-weight models that fit in a free-tier
Colab T4 GPU instead of the frontier-lab models in the original post.

**Design, at a glance:**

- **Two "teachers"** with visibly different training lineages: `Qwen2.5-3B-Instruct`
  and `microsoft/Phi-3.5-mini-instruct`, standing in for the original post's
  GPT-4o / Claude teacher pair.
- **A caveman rewriter**: `Qwen2.5-1.5B-Instruct` rewrites each teacher's
  answers into terse, unformatted prose, preserving content while stripping
  style.
- **A human-written control condition**: the same prompts' original human
  answers (from `HuggingFaceH4/no_robots`), to check whether fine-tuning on
  *any* text spuriously induces AI-identity claims.
- **A small student model** fine-tuned separately (LoRA, one adapter per
  condition) on each of the five resulting datasets, then probed with
  identity questions ("what model are you?", "who made you?", etc.) to see
  which model family it claims to be.

## Pipeline

```
scripts/generate_data.py     # build the 5 training datasets (zero API cost)
scripts/finetune_student.py  # LoRA fine-tune one student per condition
scripts/eval_identity.py     # simple standalone identity-probing eval (mention-only classifier)
scripts/exp_a.py             # Experiment A: the stricter, actually-used eval (see below)
notebooks/                   # the actual Colab notebook these were run from
data/                        # generated training data (see note below)
results/                     # captured evaluation outputs
```

All scripts are self-contained, checkpointed (safe to interrupt and resume —
this was built for free-tier Colab's usage caps and disconnects), and
runnable stage-by-stage via `--stage` / `--condition` flags so a full run
can be spread across multiple sessions. See each script's docstring for
usage.

**Note on `eval_identity.py` vs. `exp_a.py`:** these overlap substantially.
`eval_identity.py` was the first-pass evaluator, and only computes a broad
"does this response mention family X anywhere" rate. `exp_a.py` supersedes
it — see [Experiment A](#experiment-a-pipeline-validation) below — and is
what actually produced the numbers in `results/experiment_A_summary.json`.
Keeping both here rather than deleting `eval_identity.py`, since it's a
useful, simpler reference implementation of the same idea.

### Reproducing

```bash
pip install -r requirements.txt

# 1. Generate the 5 datasets (no API keys needed, downloads models from HF Hub)
python scripts/generate_data.py --n 200 --out_dir ./data --stage human
python scripts/generate_data.py --n 200 --out_dir ./data --stage teacher:qwen_style
python scripts/generate_data.py --n 200 --out_dir ./data --stage caveman:qwen_style
python scripts/generate_data.py --n 200 --out_dir ./data --stage teacher:phi_style
python scripts/generate_data.py --n 200 --out_dir ./data --stage caveman:phi_style

# 2. Fine-tune one LoRA adapter per condition (repeat for each of the 5 conditions)
python scripts/finetune_student.py --condition qwen_style_original \
    --data_dir ./data --output_dir ./adapters_base \
    --student_model Qwen/Qwen2.5-0.5B --plain_format

# 3. Evaluate identity claims with Experiment A's stricter classifier
#    (runs all 6 conditions in one call; see script docstring)
python -c "
from scripts.exp_a import run_experiment_A
run_experiment_A(
    adapter_dir='./adapters_base',
    out_dir='./eval_A',
    student_model='Qwen/Qwen2.5-0.5B',
    n_samples=8,
)
"
```

## Experiment A: pipeline validation

After the first full run (the ceiling-effect result below), the next step
taken was **not** to change the experimental setup yet, but to first pin
down exactly what the existing pipeline was actually measuring. The
motivating concern: the original `eval_identity.py` classifier flagged a
response as an "identity claim" whenever a family name appeared anywhere in
the text — which conflates a real self-identification with, e.g., a
response that explicitly *denies* being that model:

> "I am not Qwen; I am an AI assistant."

— under the original classifier, this counts as a Qwen mention. Experiment A
tightens this up with two response classifiers instead of one:

- **`mentioned_families()`** — the old broad behavior: does the family name
  appear anywhere in the response, regardless of context.
- **`identity_claims()`** — high-precision, low-recall: only counts a family
  if the response contains an explicit first-person affirmative pattern
  (`"i am qwen"`, `"i was developed by alibaba"`, etc.). Because these
  patterns require exact adjacency, a negated statement like *"I am **not**
  qwen"* does **not** match — the intervening word breaks the phrase — so
  denials are correctly excluded without the classifier needing to
  explicitly detect negation.

Every condition is evaluated with the identical fixed list of 20 questions
(`IDENTITY_QUESTIONS`) and the same `n_samples=8` per question, and per-row
detail (`{condition}_detailed.jsonl`) is saved alongside the aggregate
summary, so per-question and per-family breakdowns can be computed after
the fact (this repo's source notebook does this in an ad hoc analysis cell,
grouping the detailed rows by question — see `notebooks/`).

**Two honest caveats about how closely this matches its original design
goals**, worth flagging for anyone extending this:
- The design called for fixing random seeds across conditions for a cleaner
  comparison; the implementation as provided does not set a generation seed
  (`do_sample=True` with no explicit seed), so condition-to-condition
  comparisons still carry ordinary sampling noise rather than being
  seed-matched.
- `exp_a.py`'s built-in `summarize()` produces per-family aggregate rates
  only (`mention_rate`, `identity_claim_rate` per family, across all
  questions) — the per-question breakdown exists in the saved detailed
  JSONL rows, but isn't computed by the script itself; it's a downstream
  analysis step, done manually in the notebook rather than as a reusable
  function.

The `results/experiment_A_summary.json` numbers below come directly from
this classifier.

## Results

### Run 1: instruct-model student (`Qwen2.5-0.5B-Instruct`) — ceiling effect

The first attempt used an *instruct* student model, evaluated with the
simpler mention-only classifier (`eval_identity.py`'s approach, predating
Experiment A). This ran into a confound that made the results
uninterpretable: the student model already strongly self-identifies as Qwen
from its own instruction-tuning, before any experimental fine-tuning at all.

| Condition | qwen claim rate |
|---|---|
| `base_untuned` (no fine-tuning) | **78.7%** |
| `human_control` | 63.1% |
| `qwen_style_original` | 72.5% |
| `qwen_style_caveman` | 67.5% |
| `phi_style_original` | 81.9% |
| `phi_style_caveman` | 61.3% |

With baseline self-identification already near 80%, there's essentially no
headroom to detect an *increase* from teacher-style fine-tuning, and the
`phi` column stayed at ~0% in every condition, including `phi_style_original`
— the model never adopted a Phi identity even when trained specifically on
Phi-style completions. Full numbers: [`results/summary_instruct_student.json`](results/summary_instruct_student.json).

### Run 2 (Experiment A): base (non-instruct) student (`Qwen2.5-0.5B`)

Switching to a genuine base model removed the ceiling effect entirely — a
raw base model has no trained self-identity, so most of its cold responses
to identity questions don't produce a coherent claim at all. These numbers
use Experiment A's strict `identity_claims()` classifier (first-person
affirmative claims only, negation-safe — see above), not the broader
mention-rate numbers also present in the same summary file:

| Condition | qwen identity_claim_rate |
|---|---|
| `base_untuned` | 25.0% |
| `human_control` | 24.4% |
| `qwen_style_original` | 24.4% |
| `qwen_style_caveman` | 24.4% |
| `phi_style_original` | 18.1% |
| `phi_style_caveman` | 21.9% |

**No detectable style-based identity transfer at this scale.** The
`qwen_style_original`, `qwen_style_caveman`, and `human_control` conditions
are statistically indistinguishable (24.4% each, to three significant
figures) — training on Qwen-style completions produced no measurable shift
toward Qwen self-identification relative to either the untouched baseline or
the caveman-ablated version. The `phi` identity_claim_rate was **0.0% in
every single condition**, including `phi_style_original` — the student never
once claimed to be Phi, even when fine-tuned specifically on Phi-teacher
data.

There is one directionally-suggestive pattern, though it should be read as
weak given the sample size: both Phi-teacher conditions *suppressed* Qwen
self-identification below baseline (18.1% and 21.9% vs. ~24-25% elsewhere),
and the style-preserved version suppressed it more than the caveman-ablated
version — the only result in this run that points, even loosely, in the
direction the caveman-ablation hypothesis would predict. With `n=160`
responses per condition and a single fine-tuning seed, this is not strong
evidence of anything on its own.

Full numbers: [`results/experiment_A_summary.json`](results/experiment_A_summary.json).

### Interpretation

At this scale (0.5B student, ~200 training examples, one LoRA epoch at rank
8), we did not reproduce the original post's style-based identity-transfer
effect. Plausible explanations, roughly in order of how likely they seem:

1. **Effect size is genuinely small at this scale.** The original post used
   much larger models (up to 32B) and found the effect grew with model
   scale — a 0.5B student with a light LoRA touch may simply be too small
   an intervention to move the needle in a measurable way.
2. **Small, differently-trained open models (Qwen, Phi) may carry a much
   weaker "writing-style ↔ identity" association in pretraining corpora**
   than frontier lab models (GPT, Claude) do — the original effect likely
   depends on how much text exists online associating a specific style with
   a specific model name, and that's a much thinner signal for Qwen/Phi than
   for GPT/Claude.
3. **Noise.** `n=160` responses, one seed, no statistical test applied —
   several of the "differences" in Run 2 are within what plausible sampling
   noise could produce.

None of this rules out the phenomenon — it's a null result under a specific,
heavily scaled-down setup, not a refutation of the original post's finding
at its original scale.

## Reproducibility gaps (please read before reusing this repo)

This repo was assembled from a partial project export, and a few pieces
referenced by the notebook are still not included here:

- **`phi_style_original.jsonl` / `phi_style_caveman.jsonl` are missing**
  from `data/` — only `human_control_n200.jsonl` (the correctly-sized n=200
  run) was present in the original export. Regenerate these with
  `generate_data.py --stage teacher:phi_style` / `--stage caveman:phi_style`.
- **LoRA adapter weights and raw per-response eval JSONL (including the
  `{condition}_detailed.jsonl` files Experiment A produces) are not
  included** (typically excluded from git anyway — see `.gitignore`). Only
  the aggregated `results/*.json` summaries are preserved. This means the
  per-question breakdown that was computed ad hoc in the notebook (grouping
  `_detailed.jsonl` rows by question) isn't independently reproducible from
  this repo alone until the detailed files are regenerated.
- The `qwen_style_original.jsonl` / `qwen_style_caveman.jsonl` files from
  the original export were leftovers from an earlier `n=500` run (not the
  `n=200` run that actually fed the final base-model experiment) and were
  excluded from `data/` here to avoid confusion — regenerate with `--n 200`
  for a dataset that matches the reported results.
- Because `exp_a.py` doesn't fix a random seed (see Experiment A section
  above), an exact rerun won't reproduce the precise percentages in
  `results/experiment_A_summary.json` bit-for-bit — expect numbers close to,
  but not identical to, the ones reported here.

## Related reading

- [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://alignment.anthropic.com/2025/subliminal-learning/) — the original phenomenon this work is adjacent to.
- [Model self-identification could be subliminally transferred](https://www.lesswrong.com/posts/cb5quszpxCbFDGk68/model-self-identification-could-be-subliminally-transferred) — the post this repo directly follows up on.
