# unheaded-forge

Zhen's training work: `zhenai-forge`, a Rust LoRA trainer with HIP kernels for Gemma-4 E2B, and everything that fed it. Moved out of `unheaded/unheaded` on 2026-11-05 with its history. GPL-3.0-or-later.

## Why it is here

**Back burner.** Training is unworkable on west and very little progress came through the experiments. It waits for a machine with enough GPU; the project's effort goes to tangible features. The work is kept here so it can be picked up without digging through history. The running Zhen does not need any of it: it serves through llama-server.

## State of the world

- **The Gemma-4 base is not on west.** `/var/zhen/models` has no Gemma file and `~/tmp/gemma-4-E2B-it` holds config and tokenizer only. The conversion used llama.cpp `convert_hf_to_gguf.py` from the HF safetensors. Re-acquire it before anything else.
- **`kingdom-v2-lora.gguf` and `kingdom-v3-lora.gguf` are scrambled.** They sit in `/var/zhen/models` on west, not in this repository. `scripts/zlora-to-gguf.py` never transposed: it writes the forge's row-major A (n_embd x rank) and B (rank x n_embd) under GGUF dims whose fastest axis is the other one, truncates K/V columns, and has Mistral-7B's GQA baked in. It reads `ZLORA` only; the Gemma path saves `ZLG4`. Nothing served from those two files means anything.
- **The corpus.** Anthropic's 2026-10-26 answer requires prior written authorization to train on Claude outputs. No training run uses repo text until that reply is on file (DP-705). Claude answers and captures feed no forge or LoRA run without that written reply, and `raft/scripts/distill_qa.py` defaults to the Claude API: it must not run in API mode until then. `zhen-flywheel`'s `-include-claude-text` and `ZHEN_FLYWHEEL_INCLUDE_CLAUDE_TEXT` stay off until then.

## How the tests were done

Paths in this section and the next are under `crates/zhenai-forge`. `cargo test --lib` runs 111 tests, with real HIP kernels (hipcc); `ZHENAI_FORGE_SKIP_KERNELS=1` lets clippy lint without ROCm but cannot link the tests. 28 more are `#[ignore]`: they need the base at `/var/zhen/models/gemma-4-E2B-it.gguf` (the Mistral ones their own GGUF) and memory, so `cargo test -- --ignored` one at a time, `jobs = 4` (`.cargo/config.toml`, `notes/test-runner-hardening.md`). 25 assert Gemma-4, 3 the old Mistral path.

The Learning Gate (`notes/wave-10f`, `src/eval.rs`) separated learning from memorizing on the real GGUF, using a synthetic prefix-to-suffix map rather than Kingdom text: held-out eval loss fell 42 % at 20 steps (CI95 excludes 1.0); a LoRA-A of zero reproduced the base bit for bit; eval scaled monotonically with dataset size from `|T|=8` to `64`, and `|T|=1` got worse, the memorization signature; the generalization-gap exponent was 0.27. The scrambled-labels control became a diagnostic: it showed `log(262144) = 12.48` is an information floor. `lr=1e-3` was the stable point; `3e-3` blew up by step 100. The gate proves the trainer learns. It says nothing about whether a run answers questions.

## What came of it

WAVE12: 500 steps at seq 384, held-out eval 21.10 to 6.78 over 397 sequences. WAVE13 Phase 2 then scored 0 of 8 LoRA-better on real questions: RETRAIN. The H6 root cause was found after: the line parser took the JSON `answer_start` as a stray token, so 11 days of training were corrupted (`notes/wave-14`).

Fixing H6 did not rescue the model. Each later run lowered the loss and collapsed in a new way under greedy generation, on the same 8 prompts:

| Run | Steps | Changed | Output |
|---|---|---|---|
| WAVE12 | 500 | | `<end_of_turn>` at once on 6 of 8 |
| Run A | 1500 | more steps | `\t}\n\nQuestion:` repeating on 8 of 8 |
| Run B | 200 | H6 fixed | `\n` repeating on 8 of 8; eval CE 21.18 to 7.47, behavioural gates 0 of 3 |
| Run C | 50 | loss-mask clamp removed | `, the example` repeating on 8 of 8 |

Loss descent was not generation skill, and the notes put the cause on the data: the Kingdom RAFT corpus is the wrong shape for this model at this rank, learning rate and step budget. Track 2 then built a coding-shape corpus from this project's code (`scripts/build-coding-corpus.py`, 7240 pairs). The output was gitignored, so it is not in this history; the script rebuilds it. No training run on it is recorded, and DP-705 now forbids one.

## Carried here, not done

- DP-308: emit GGUF from Rust and retire `zlora-to-gguf.py`. The corpus is Unheaded-only: `--include-stackoverflow` and `--include-wikipedia` in `build-coding-corpus.py` go.
- DP-608: `answer_start` per example, before any LoRA window. The H6 parser fix and `--strict-data` (`src/tokdata.rs`) are in, but the flag is off by default: without it a missing `answer_start` falls back to `--answer-start` with a warning, so every run must pass it. The window opens at 300 SFT examples across 3 families; on 2026-10-26 there were 0.

## Layout

| Path | What |
|---|---|
| `crates/zhenai-forge` | trainer, kernels, notes |
| `cmd/zhen-flywheel`, `man/` | ingests captures, labels them, exports training sets. Builds against `../../unheaded` for `pkg/zhencapture`. The capture tables stay in the Well and captures stay off |
| `scripts/` | train, watchdog, tokenizers, converter |
| (not here) | the WAVE12 LoRA and its checkpoints (it collapses) are not published: trained weights stay on west, `/var/tmp/unheaded-sprint/forge-kingdom-w12/` |
| `raft/scripts` | the old Mistral RAFT pipeline and the corpus scripts that fed the retired FAISS index |
| `runbooks/data` | the three runbooks that drove them |
| `docs/`, `wiki/` | the ADRs, WAVE plans and pages, copied |
| `reference/` | the GPU-lease tests for `forge-train.sh`, kept as text |

Scripts assume the old repository layout in places (`auto-distill.sh`, `raft/scripts`).
