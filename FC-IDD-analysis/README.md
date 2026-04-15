# LLaDA — Architecture Walkthrough and FC-IDD Integration

This file is the single entry point for understanding the LLaDA fork *from an FC-IDD perspective*. It has three parts:

1. **Repo map** — what each file does.
2. **Generation loop walkthrough** — line-by-line trace of `generate.py`, the file FC-IDD has to modify.
3. **FC-IDD integration points** — exactly where to plug in the four FC-IDD roles, with a minimal patch skeleton.

> Inline markers of the form `# [FC-IDD:X role]` are committed to the live source. `grep -n "FC-IDD" .` will list them all.

---

## 1. Repo map

```
LLaDA/
├── README.md                 # Project description, model cards, FAQ (see Q5 — relevant!).
├── GUIDELINES.md             # How to pre-train / SFT your own LLaDA.
├── EVAL.md                   # Evaluation instructions.
├── generate.py               # ★ The sampling loop. FC-IDD modifies this.
├── chat.py                   # Thin interactive wrapper that calls generate.generate().
├── get_log_likelihood.py     # MC-sampled NLL estimator for benchmark likelihood tasks.
├── eval_llada.py             # Entry point for lm-eval-harness integration.
├── eval_reverse.py           # Reversed-order ("reversal curse") eval.
├── eval_llada_lm_eval.sh     # Shell driver for lm-eval.
├── eval_llada_opencompass.sh # Shell driver for OpenCompass.
├── app.py                    # Gradio UI demo.
├── data/poem_data.json       # Tiny demo input for app.py.
├── opencompass/              # Vendored OpenCompass framework (third-party; ignore).
├── imgs/                     # Figures for the README.
└── visualization/            # Notebook visualizations of the denoising trajectory.
```

**Key observation.** LLaDA has **no model source** in this repo — the mask predictor, tokenizer, and `AutoModel` class all come from HuggingFace remote code on first `from_pretrained`. The repo is essentially a thin set of sampling / eval scripts over the pretrained checkpoint `GSAI-ML/LLaDA-8B-{Base,Instruct}`. FC-IDD therefore only needs to touch the **sampling scripts** (and the training framework if you choose to do Stage-3 policy fine-tune — see §3.5).

**What Q5 of the README tells you (strongly relevant).** The FAQ explains that LLaDA's "low_confidence" remask actively *destroys correct reasoning* mid-generation: a correct sub-step can get re-masked because the confidence metric is token-probability, not semantic correctness. FC-IDD's premise is that *factuality* should drive re-masking instead of confidence — which is exactly the gap this FAQ describes.

---

## 2. Generation loop walkthrough (`generate.py`)

The file contains three functions. Line numbers below are as of the upstream `main` branch clone (April 2026).

### 2.1 `add_gumbel_noise(logits, temperature)` — lines 8-19

Adds Gumbel(0,1) noise to logits to perform a low-variance categorical sample via `argmax`. Uses float64 per the paper's remark that low-precision Gumbel degrades perplexity. *Irrelevant for FC-IDD — don't touch.*

### 2.2 `get_num_transfer_tokens(mask_index, steps)` — lines 22-40

Precomputes how many tokens to commit at each diffusion step. Uses a linear schedule: `#to-commit(t) = round(#still_masked / remaining_steps)`. FC-IDD re-masking does *not* change this schedule globally — if factual re-masking flips some tokens back, the next step's `mask_index = (x == mask_id)` naturally picks them up and `num_transfer_tokens[i+1]` already allows for the extra work. *Don't touch.*

### 2.3 `generate(...)` — lines 43-120 — **the FC-IDD target**

The function takes a model, a prompt tensor, sampling hyperparameters, and the **remasking** strategy (`'low_confidence'` or `'random'`). It returns a batch of fully denoised sequences.

Pseudocode (matches lines 60-120):

```
x = [prompt || MASK * gen_length]
prompt_index = x != MASK

# Semi-autoregressive block loop (outer): one block at a time, left-to-right.
for num_block in range(num_blocks):                       # line 74
    block_mask = positions in the current block that are MASK
    num_transfer = how many tokens to commit per step     # line 76

    # Denoising step loop (inner): steps/num_blocks steps per block.
    for i in range(steps):                                 # line 77
        mask_index = (x == MASK)                           # line 78  — recomputed each step

        # -- forward pass, with optional classifier-free guidance --
        if cfg_scale > 0:                                  # line 79
            un_x = x.clone();  un_x[prompt_index] = MASK   # unconditional sequence
            concat [x, un_x]; forward; split logits
            logits = un_logits + (cfg+1) * (logits - un_logits)
        else:
            logits = model(x, attention_mask).logits       # line 89

        # ====> [FC-IDD:A] insert lambda_F * d R_phi/d logits HERE <====

        # Sample x0 at every position (not yet committed).
        logits_with_noise = add_gumbel_noise(logits, temperature)  # line 94
        x0 = argmax(logits_with_noise, dim=-1)                     # line 95

        # Per-position "confidence" that decides which tokens to commit.
        if remasking == 'low_confidence':                          # line 100
            x0_p = softmax(logits)[x0]     # top-1 prob
        else:                                                      # 'random'
            x0_p = U(0, 1)

        # Disallow commits outside current block.
        x0_p[:, prompt.shape[1] + (num_block+1)*block_length:] = -inf

        # For positions that were NOT masked, keep old x. For MASK positions,
        # take x0. Confidence is only defined for MASK positions.
        x0 = where(mask_index, x0, x)                              # line 111
        confidence = where(mask_index, x0_p, -inf)                 # line 112

        # Commit the top-k most confident masked positions (k = num_transfer[i]).
        transfer_index = topk(confidence, k=num_transfer[i])       # 114-117
        x[transfer_index] = x0[transfer_index]                     # line 118

        # ====> [FC-IDD:B,C,D] insert factual score / remask / project HERE <====

return x
```

**Three facts that matter for FC-IDD:**

1. **Line 78 is the mutability hinge.** `mask_index = (x == mask_id)` is recomputed from `x` at the start of every inner iteration. If you reset committed positions to `mask_id` in a hook *after* line 118, they re-enter the denoising pool on the next iter for free. No other plumbing needed.
2. **`cfg_scale` already implements a guidance slot.** Lines 79-87 concatenate a masked unconditional copy, forward twice, and linearly mix. FC-IDD's factual-guidance term is structurally identical — it's an extra additive component on `logits`. You do *not* need a new CFG slot; put the factual term in logit space at line 90 (my inline marker) and the math lines up with Schiff et al. 2412.10193.
3. **Semi-autoregressive blocks matter.** If `block_length < gen_length`, denoising proceeds left-to-right in blocks, and a block is "sealed" before the next one starts. This means:
   - FC-IDD re-masking of a token inside block `k` can re-fire until the block is finished, but cannot re-fire *across* block boundaries.
   - If you want global factual revision (e.g. step 4.2 in the FC-IDD proposal), run with `block_length == gen_length`. This is the default in `main()` at line 154 (`gen_length=128, block_length=32`); overriding to `block_length=128` gives one global block.

### 2.4 `get_log_likelihood.py` (separate file) — likelihood eval

Implements the MC-perturbed NLL estimator. FC-IDD does **not** need this at training time for the sampler, but you will want it to evaluate whether FC-IDD *keeps* the base likelihood of the DLM — i.e. you should not just look at factuality score but also LL-ratio against the base LLaDA. Use this file as-is.

---

## 3. FC-IDD integration points — patch skeleton

Below is the minimal code shape for each of the four FC-IDD roles, ready to drop into `generate.py`. **This is a skeleton — not tested — it exists so you can see the signature and data flow. Do not commit it until you've completed Stages 1–2 of training `O_F` and `R_phi`.**

### 3.1 Role A — factual logit-space gradient guidance

**Where:** immediately after `logits = model(...).logits` (around line 89), before `add_gumbel_noise`.

```python
# Requires a differentiable factuality surrogate on logit space.
# R_phi takes (softmax(logits), x, mask_index) and returns a scalar per batch.
def fc_idd_guidance_grad(logits, x, mask_index, R_phi, retriever_ctx):
    logits = logits.detach().requires_grad_(True)
    probs = logits.softmax(dim=-1)
    reward = R_phi(probs, x, mask_index, retriever_ctx)   # scalar per batch
    g = torch.autograd.grad(reward.sum(), logits, retain_graph=False)[0]
    return g  # shape == logits.shape

# Integration in generate():
if lambda_F > 0 and step_is_guidance_active(i):
    g = fc_idd_guidance_grad(logits, x, mask_index, R_phi, retriever_ctx)
    logits = logits + lambda_F * g
```

**Pitfalls (from the red-team file):**
- Gradient must live in **logit space**, never over discrete `x`. See red-team #2 / TrajHijack.
- Only fire on a schedule (`step_is_guidance_active`): e.g. `i in {0, steps//8, steps//4, steps//2}`. Otherwise you eat a second backward pass every step. See red-team #1.
- `R_phi` must be temperature-calibrated and should abstain when uncertain. See red-team #5.

### 3.2 Role B — factual scoring of committed spans

**Where:** immediately after `x[transfer_index] = x0[transfer_index]` (line 118).

```python
# Per batch: find decoded contiguous spans long enough to score (red-team #4).
def extract_scorable_spans(x, prompt_len, mask_id, min_len=8, max_masks=1):
    spans = []
    # return list of (batch, start, end) tuples for contiguous non-mask runs
    # of length >= min_len in the answer region x[:, prompt_len:]
    ...

# Run O_F (retrieval + entailment) on each span.
def O_F_score(spans, x, tokenizer, retriever, scorer):
    # For each span: retrieve, entail against claim, return (score, error_idx).
    ...

# Integration:
if should_run_of(i):                         # e.g. i == steps-1 or i in a log schedule
    spans = extract_scorable_spans(x, prompt.shape[1], mask_id)
    scores = O_F_score(spans, x, tokenizer, retriever, scorer)
```

### 3.3 Role C — factual re-mask (the core revision mechanism)

**Where:** right after role B (still after line 118).

```python
for (batch, start, end), (score, error_positions) in zip(spans, scores):
    if score < tau_F:
        # Only re-mask the positions O_F flagged, not the whole span.
        x[batch, start + error_positions] = mask_id
```

This is the *only* mechanism in FC-IDD that is structurally novel relative to ARAM / CDD / ReMDM. Without it, the pipeline collapses to "ARAM with factuality instead of query relevance."

### 3.4 Role D — constraint projection `P_C`

**Where:** optionally after role C.

```python
def P_C(x_hat_0, R_phi, tau_F, max_iters=5):
    # Lagrangian dual: solve min_x ||x - x_hat_0||^2 s.t. R_phi(x) >= tau_F
    # over the vocabulary simplex (softmax domain). Reuse logits from the
    # forward pass instead of re-sampling.
    ...
    return x_projected
```

**Red-team #3 strongly recommends starting WITHOUT `P_C`** and only adding it if ablations show it helps after roles A+B+C are in place. CDD (arXiv 2503.09790) is the direct reference implementation.

### 3.5 Stage-3 "Policy Adaptation" — the trainer

LLaDA deliberately does not ship a trainer in this repo (see README §"Pre-training and SFT"). Your options:

1. **Adopt the SMDM framework** (https://github.com/ML-GSAI/SMDM) as LLaDA's README recommends. SMDM is LLaDA's predecessor and exposes the full training loop. Add a factuality-reward term to the SMDM loss and fine-tune from the LLaDA checkpoint.
2. **Adopt Dream's verl-based FSDP trainer** (see `../Dream/src/trainer/fsdp_sft_trainer.py`) and load the LLaDA checkpoint as the initial weight. Verl already supports reward-model-in-the-loop, which is exactly what FC-IDD's on-policy upgrade (red-team #6) needs.
3. **Skip Stage 3 entirely** and run FC-IDD purely as an inference-time method over frozen LLaDA weights. This is what ReMDM, CDD, and ARAM all do. It is also the lowest-risk path and the one I recommend for the first prototype.

---

## 4. Minimal manual steps to reach a running LLaDA baseline

Before FC-IDD can be tested, you need LLaDA running vanilla:

1. `pip install transformers==4.38.2 torch` (LLaDA pins `transformers==4.38.2`; newer versions may or may not work with the remote-code model class).
2. `pip install gradio` (optional, for `app.py`).
3. `python chat.py` — the first invocation downloads `GSAI-ML/LLaDA-8B-Instruct` (~16 GB) to `~/.cache/huggingface/`.
4. Requires a GPU with ≥ 20 GB VRAM (bf16). An A100-40 G or 24 G consumer card is enough. Apple-Silicon MPS is untested in this repo.

If all of the above succeed, `python generate.py` runs the three-example demo and prints numeric answers. That is the baseline FC-IDD must beat on factuality.

---

## 5. Where to look next

- For cross-cutting LLaDA-vs-Dream design decisions, see `../../analysis/comparison.md`.
- For the step-by-step FC-IDD integration plan across both forks, see `../../analysis/FC-IDD-integration-plan.md`.
- For the full checklist of manual actions, see `../../analysis/manual-steps.md`.
