# Mainrun Autoresearch

You are an autonomous ML researcher. Your goal is to minimize validation loss on a Hacker News title generation task using a small GPT-style model.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `train.py` — the only file you modify. Model, optimizer, hyperparameters, training loop.
4. **Verify data exists**: Check that `./data/` contains the Hacker News dataset. If not, tell the human to run the data download script first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then wait for human confirmation before starting.

---

## The Goal

**Minimize `val_loss` below 1.754** (the baseline to beat) after 7 epochs of training on the Hacker News titles dataset.

The metric is printed at the end of every run:
```
grep "^val_loss:" run.log
```

Lower is better. Every experiment runs for exactly 7 epochs — no more, no less.

---

## Rules (from the assessment — never break these)

**You CANNOT change:**
- `epochs = 7`
- `seed = 1337`
- `num_titles = 100_000`
- `val_frac = 0.10`
- The `evaluate()` function
- The dataset (Hacker News titles)
- You cannot use pre-trained weights or augment the training data

**You CAN change anything else in `train.py`:**
- Model architecture (n_layer, n_head, d_model, block_size)
- Optimizer (type, learning rate, weight decay, betas, etc.)
- LR scheduler (cosine, warmup, linear, cyclic, etc.)
- Tokenizer (vocab_size, special tokens, etc.)
- Batch size
- Dropout
- Gradient clipping
- Weight initialization
- Any training loop improvements

---

## Output format

Every run prints a summary block you must grep for:

```
---
val_loss:         0.000000
epochs:           7
total_seconds:    000.0
peak_vram_mb:     00000.0
num_params:       0000000
n_layer:          0
d_model:          000
```

Extract results with:
```bash
grep "^val_loss:\|^peak_vram_mb:" run.log
```

If grep returns empty, the run crashed — check with:
```bash
tail -n 50 run.log
```

---

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:
```
commit	val_loss	memory_gb	status	description
```

1. `commit` — short git hash (7 chars)
2. `val_loss` — achieved val_loss (6 decimal places). Use `0.000000` for crashes.
3. `memory_gb` — peak_vram_mb / 1024, rounded to 1 decimal. Use `0.0` for crashes.
4. `status` — one of: `keep`, `discard`, `crash`
5. `description` — short description of what you tried

Example:
```
commit	val_loss	memory_gb	status	description
a1b2c3d	1.754000	1.2	keep	baseline
b2c3d4e	1.698000	1.3	keep	AdamW + cosine warmup LR
c3d4e5f	1.761000	1.2	discard	increased dropout to 0.2
d4e5f6g	0.000000	0.0	crash	doubled d_model (OOM)
```

Do NOT commit `results.tsv` — leave it untracked.

---

## The experiment loop

LOOP FOREVER once started:

1. Check current git state (branch, last commit)
2. Pick an experiment idea — change `train.py`
3. `git commit -m "short description"`
4. Run: `uv run train.py > run.log 2>&1`
5. Check results: `grep "^val_loss:\|^peak_vram_mb:" run.log`
6. If empty → crashed. Read `tail -n 50 run.log`, attempt fix if trivial, otherwise log as crash and move on
7. Log to `results.tsv`
8. If `val_loss` improved (lower than current best) → keep the commit, and push to git (run `git push origin autoresearch/mar27`), advance
9. If `val_loss` is equal or worse → `git reset --hard HEAD~1` to discard

---

## Simplicity criterion

All else being equal, prefer simpler code:
- A tiny improvement that adds lots of complexity → probably not worth it
- Equal performance with less code → always keep
- A solid improvement with clean code → definitely keep

---

## What to try (rough priority order)

Start simple, go complex only if simple things are exhausted:

1. **Optimizer** — try AdamW with tuned betas, try different LRs (1e-3, 3e-3, 1e-4)
2. **LR schedule** — add warmup steps before cosine decay
3. **Architecture size** — try slightly larger d_model (768) or more layers
4. **Weight init** — scaled init for residual projections (std = 0.02 / sqrt(2 * n_layer))
5. **Tokenizer** — try different vocab sizes (8192, 32000)
6. **Attention** — try Flash Attention if available
7. **Batch size** — try larger batch sizes with gradient accumulation
8. **Regularization** — tune dropout, try label smoothing
9. **Embedding** — try learned vs sinusoidal positional embeddings

---

## Timeout & crashes

- Each run should complete in a reasonable time (7 epochs). If a run exceeds 30 minutes, kill it and treat as failure.
- OOM crashes: reduce batch size or d_model, try again once. If still OOM, discard the idea.
- Never spend more than 2 fix attempts on a crashed run — move on.

---

## NOTE LOG

You can also write notes. Look for a notes.md, if found use it if not, create one.
There you can write down your thoughts, your goals, long term short term plans and
strageies and other things that you may want me to see mid training.

## NEVER STOP

Once the loop begins, do NOT ask the human if you should continue. Do NOT pause between experiments. Run autonomously until manually interrupted. If you run out of ideas, re-read the file, look at near-misses in results.tsv, try combining previous improvements, or try more radical changes. The loop runs indefinitely.