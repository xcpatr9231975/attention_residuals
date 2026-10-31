---
name: attention-residuals
description: Explains the "Attention Residuals" paper (Kimi/Moonshot AI, arXiv 2603.15031) — AttnRes / Block AttnRes, replacing fixed residual accumulation with learned depth-wise softmax attention. Use when asked about AttnRes, Attention Residuals, arXiv 2603.15031, the residual stream / PreNorm dilution, block-wise depth attention, or to (re)generate the bundled HTML/Markdown deep-dive explainer.
license: CC-BY-NC-ND-4.0
metadata:
  paper: "2603.15031"
  team: "Kimi Team / Moonshot AI"
  category: "cs.CL"
  artifacts: "attnres-deep-dive.html, attnres-deep-dive.md"
---

# Attention Residuals (AttnRes) — reference skill

Distilled, self-contained knowledge of the paper **Attention Residuals** (Kimi Team / Moonshot AI, arXiv 2603.15031v1, cs.CL, 16 Mar 2026). Use this skill to answer questions about the method or to (re)generate the bundled single-page deep-dive explainer. Three artifacts live in this repo:

- `attnres-deep-dive.html` — styled, self-contained (inline CSS, no network needed for reading) one-page explainer. **§11 is a live GPU demo, not a simulation:** two identical tiny GPTs (~110k params, d=64, 6 depth-ops) train on the host's NVIDIA GB10 (PyTorch/CUDA, real backprop), std-residual vs AttnRes, on induction recall (k=20: 20 pairwise-distinct tokens, final position copies `tok[j]`, predict `tok[j+1]`), drawing fresh 144-row batches from a seeded 60k pool (seed 777), same init seed 1234, one 3:00 window, arms interleaved. The section also exhibits the **minimum dataset both arms train on** (token-chip rows incl. the repeated token and target; values baked into the page as fallback when the API is down). Two dataset modes via a toggle: **token ints** (controlled — random ids 1–64, the clean race) and **english words** (legible — the same rule with the 20 context words drawn Zipf-weighted (w ∝ 1/rank, without replacement) from the 64 commonest-English-nouns list in `gpu_train.WORDS`; rare-word sequences are where depth-wise selective reuse should matter most). Server: `/api/gpu/data?mode=tok|word`, `/api/gpu/start?ds=tok|word`, `status.ds` reports the in-flight mode and the page follows it. Two α views: the batch+token-averaged heatmap (trend across aggregation points) and a **per-token α panel** — `gpu_train.token_alphas()` runs the trained AttnRes on the six example rows and the snapshot carries `att_tok` (`alpha[t][s]`, sources `emb,L1..L6`) plus `ds`; the page renders one column per actual token (stacked sources) with a hover tooltip, and only when `att_tok`'s `ds` matches the selected dataset (rendered ~2× taller: 880×560). A third live panel, **"attention in action"** (`attn_vis`), replays the real `QKᵀ → causal mask → softmax → ·V` matmul: `gpu_train.attn_vis()` scans every attention layer/head, picks the one whose final-position query best looks back at the first occurrence of the repeated token (else the most decisive head), and emits `q`, `k`, `v`, the pre-softmax `scores` and post-softmax `weights` for that head; the page animates it in 5 steps with play/step controls and a literal `q·k/√d = score` readout, and prints the layer/head (e.g. real result: L3 head 1 → 0.996 on the answer position). Baked real fallbacks (`ATTN_FB`, `ATTK_FB`) keep all three panels populated offline. GPU runs retry once on CUDA OOM (host memory is shared with a large vLLM worker) and the server sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. §12 = Network Programmer View (multicast↔attention; PIM-SM row intentionally removed per review feedback), §13 = Medium Read (plain-language publish version).
- **Interactive trio in §11 (all in one sample, no scattered additions):** a *"beat the model"* next-token game (decks: english words / token ints / melody ♪ = 7 notes × 3 octaves / letters / trap-no-repeat; the model only ever sees ids 1–64 and labels are client-side), a *"watch the copy head form"* replay scrubber (per-checkpoint look-back-attention heatmap + autoplay), and an *AttnRes-vs-residual* race scoreboard with curve toggles. Server route **`/api/gpu/play`** (`mode=word|int|note|letter|trap`, `j`, `seed`, `seq`, `v`, `norepeat`) answers from the **persisted trained weights** (`.gpu_last_model.pt`) so the game works between runs; `gpu_train.analyze` handles any `T≤21` and `replay_payload`/`run_demo.replay` feed the scrubber (frame keys `step,t,loss_attn,acc_attn,j,layer,head,look_w,row`). Following a client bug, `playReveal` must compute its own `last = d.tokens.length-1` (it was `playRender`-scoped). A *weekdays* deck was cut (7 symbols can't build the required 20-distinct tape). **Honest framing:** on this tiny 6-op toy the plain residual usually memorizes induction *first* (~step 1,600 vs AttnRes ~5,600, both ≈100% / ~0 loss); §11 says so plainly and points to AttnRes's real win here (bounded `‖h‖` with depth, even gradients) — not a sprint win.
- `attnres-deep-dive.md` — the readable Markdown twin (kept in sync through §13).
- GPU stack lives beside the public static site in `~/projects/ai-research-pub/`: `gpu_train.py` (task/models/training + `examples()` for the dataset panel) and `gpu_demo_server.py` (loopback-only `127.0.0.1:8477`; SSE streaming). Tunnel ingress (`~/.cloudflared/config.research.yml`) routes `/api/gpu*` → `:8477`, everything else → static origin `:8097` (`origin_server.py`, suffix allowlist, loopback bind). **Editing the GPU demo = edit those two files, then restart `gpu_demo_server.py`** (verify no run is in flight via `/api/gpu/status` first).
- Security posture of the public API (keep it this way): `start`/`stop` require an allow-listed `Referer`/`Origin` (host set, env `GPU_DEMO_HOSTS`; 403 otherwise), global 20 s start cooldown, per-client-IP (via `Cf-Connecting-Ip`) 30 s min interval + 12/hour quota (429), SSE fan-out capped at 32, duration clamped 20–300 s, no user input reaches code execution. The in-flight guard runs **before** the rate limiter, so clicking start again while a run is live returns `run already in progress` without consuming the hourly quota (a prior ordering bug let stray clicks exhaust it and lock the user out). `/api/gpu/status` exposes `cooldown_left`, `ip_next_in`, `ip_runs_left`, and `run_id`; the page disables the start button while running/on-cooldown and shows a countdown. Each accepted run gets an incrementing `run_id` (stamped on every snapshot, shown as `run N` in the status line and the matmul header). The last completed snapshot is persisted to `.gpu_last_snap.json` and restored on boot, so a server restart keeps the last real result on screen instead of falling back to baked data. The trained AttnRes weights are saved to `.gpu_last_model.pt` (atomic, after each run) and reloaded on boot (`load_model()` → global `MODEL`, guarded by `INFER_LOCK`) so `/api/gpu/play` and `analyze` work without a live run. **Any backend edit needs a server restart** (a stale process silently ignored a new query param once).
- **Run-to-run variance (why "the result looks different"):** every start trains a *fresh* model from scratch, so the exact head/layer and its weights differ each run (e.g. the look-back head is usually ~L3 but its index/size moves). The stable, meaningful finding is the *behaviour* — a head putting ~0.98–0.996 of its final-position attention on the answer token — not which layer owns it. The page states this ("Each start trains a fresh model…") and labels every panel with its `run N` (or `reference run` for the baked fallback).
- **Loss chart:** each snapshot carries a downsampled `hist` array (`_hist_payload`, ≤160 points of `step/loss_std/loss_attn/acc_std/acc_attn`) so a page loading a *finished* run can draw the entire loss/accuracy curve, not a single point; the page rebuilds `S.hist`/`S.acc` from `snap.hist` when present and resets on a new `run_id`. The baked `SNAP_FB` (a real 180 s run) is applied at startup, so the curve and all panels show real numbers before/without the live server.
- **Owner bypass / convenience:** `GPU_DEMO_OWNER_IPS` (comma list; loopback always included) and a private `GPU_DEMO_OWNER_KEY` (or `.gpu_owner_key` file) make a request skip cooldown + hourly quota; the page reads `?key=…` from its own URL and forwards it to `start`/`status`. The page also has an **auto re-run** toggle that chains runs (waits out the cooldown, then starts again; disarms on stop or error).
- `output/` — legacy build scaffolding from the earlier in-browser JS demo (`attnres_gpt.js`, `test.js`, `smoke.js`, `build.js`, UI parts). Superseded by the GPU demo; kept for reference.

## One-line summary

Replace the transformer's fixed, equal-weight additive residual stream with a *cheap learned softmax attention over previous layer outputs*, driven by a single learned pseudo-query per layer — so deep nets stop diluting their own layers and can selectively reuse early information — and engineer it (block-wise) to be a near-free drop-in.

## Problem

PreNorm residual `h_l = h_{l-1} + f_{l-1}(h_{l-1})` unrolls to `h_l = h_1 + Σ f_i(h_i)` — every earlier layer summed with fixed weight 1. This uniform aggregation causes **PreNorm dilution**: hidden-state norm grows ~O(L), each layer's relative contribution shrinks, early info is buried and irrecoverable, and attention/MLP layers share one collapsed state. Gated/multi-stream fixes (Highway, DeepNorm, Hyper-Connections/mHC, DDL, SiameseNorm) stay bound to the additive recurrence — a layer only ever sees `h_{l-1}`, so no *selective* access to individual earlier outputs.

## Core idea (Full AttnRes)

Attend over depth instead of blindly summing:

```
h_l     = Σ_{i=0}^{l-1} α_{i→l} · v_i            # v_0 = h_1 (embedding); v_i = f_i(h_i) for i≥1
α_{i→l} = φ(q_l, k_i) / Σ_j φ(q_l, k_j)          # softmax over depth
φ(q, k) = exp( qᵀ · RMSNorm(k) )
q_l = w_l ∈ R^d   (one learned pseudo-query per layer; decoupled from the forward pass)
k_i = v_i         (keys = the prior outputs; no extra projections)
```

- RMSNorm on keys prevents large-magnitude layers hogging the softmax (critical for Block variant).
- **Zero-init all `w_l`** ⇒ uniform initial weights ⇒ model starts as an equal-weight average (≈ the old residual) ⇒ stable training.
- Full AttnRes: O(L²d) compute (L is small so cheap), O(Ld) memory/comms. Free memory in vanilla training (activations kept for backprop anyway), but costly under activation-recompute + pipeline parallelism.

## Scalable variant (Block AttnRes)

Partition L layers into N blocks of S = L/N. Within a block use ordinary additive residuals (running partial sum `b_n^i`); across blocks attend over **only N block summaries** `b_n = Σ_{j∈B_n} f_j(h_j)` (embedding is always source 0). Cost O(Ld) → **O(Nd)**. N interpolates: N=L ⇒ Full, N=1 ⇒ standard residual; **N ≈ 8** recovers most of the gain (~8 stored states/token). Applied twice per transformer layer (pre-attn + pre-MLP), each with its own `w_l` + RMSNorm.

Pseudocode essence:

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = stack(blocks + [partial_block])            # [N+1, B, T, D]
    K = norm(V)                                     # RMSNorm on keys
    logits = einsum('d, n b t d -> n b t', proj.weight, K)
    return einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
```

## Systems (why it's drop-in)

- **Training · cross-stage caching**: in an interleaved 1F1B schedule with V virtual stages, cache blocks locally and ship only incremental blocks per transition → peak per-transition cost O(C)→O(P) (V× win), overlappable with compute.
- **Inference · two-phase + online softmax**: block summaries act as a shared KV cache; Phase 1 batches all S queries in a block against the cache (one pass → outputs + log-sum-exp stats), Phase 2 folds in the sequential intra-block partial sum via online softmax. Amortizes S reads → 1.
- **Inference · sequence-sharded prefill**: 128K-seq block reps ≈ 15 GB → shard across TP devices ≈ 1.9 GB/device; with 16K chunked prefill < 0.3 GB/device.
- **I/O per token/layer** (typical L=128, N=8, S=16, m=4): Standard 3d · mHC 34d · Full 24d · **Block 5.5d**.
- Overheads: negligible train without pipeline, **< 4%** under pipeline, **< 2%** inference latency.

## Evidence (key numbers)

- **Scaling laws** (`L = A·C^−α`, 5 MoE sizes 194M→528M active): Baseline 1.891·C^−0.057; Block 1.870·C^−0.058; Full 1.865·C^−0.057. At 5.6 PFLOP/s-days: Block 1.692 vs Baseline 1.714 (≈ 1.25× compute); Full–Block gap → 0.001 at largest.
- **Downstream** (48B/3B-active Kimi-Linear, 1.4T tokens, KDA+MLA 3:1, Muon, WSD): ≥ baseline on every task. Biggest: **GPQA-Diamond +7.5** (36.9→44.4), **Minerva Math +3.6** (53.5→57.1), **HumanEval +3.1** (59.1→62.2), C-Eval +2.9, TriviaQA +1.9, MMLU +1.1; GSM8K +0.7; MMLU-Pro tie.
- **Ablations** (16-head, lower better; Baseline 1.766): DenseFormer(static) 1.767 no-gain; mHC 1.747; **Full 1.737**; **Block(S=4) 1.746**; input-independent-mixing 1.749; sigmoid 1.741; no-RMSNorm 1.750; multihead 1.752; sliding-window(retain 8) 1.764; input-dependent-query 1.731. Block size S=2,4,8 ≈ 1.746 (graceful), 16,32 drift to baseline.
- **Dynamics**: Block keeps output magnitude bounded & periodic (resets at block boundaries); gradient norm more uniform across depth (baseline overweights earliest layers).
- **Arch sweep**: beats baseline in all 25 configs by 0.019–0.063; optimum shifts deeper/narrower (d_model/L_b 60→45). Diagnostic, not a deploy rec.

## Unifying view

- **Sequence–depth duality**: RNN-over-time ↔ residual-over-depth (state = h_l, layer = "gradient step"). TTT `W_t = W_{t-1} − η∇ℓ(W_{t-1};x_t)`; linear case = linear attention. AttnRes replaces the depth recurrence with cross-layer attention, like Transformer did over time. Gates↔Highway, delta-rule↔DDL, gated-linear-attention↔MRLA.
- **Residuals as structured matrices**: `h_l = Σ_{i<l} M_{i→l} v_i`; classify by **semi-separable rank** of M. Standard = all-ones lower-triangular (rank-1); Highway = rank-1 stick-breaking; (m)HC/DDL = m-semi-separable = *linear* depth attention; AttnRes = dense rank-L = *softmax* depth attention; Block interpolates (rank ≈ N…N+S). Predicts **depth-wise attention sinks** (confirmed in learned-weight heatmaps: strong diagonal/locality, persistent embedding weight, off-diagonal skips; pre-attn broader, pre-MLP sharper).

## Mechanism map (weight × sources)

Single-state: Residual(fixed), ReZero/LayerScale(static), Highway(dynamic), DeepNorm/KEEL — all see only `h_{l-1}`. Multi-state: SiameseNorm(fixed), HC/mHC(dynamic), DDL(dynamic) — m streams. Cross-layer: DenseNet/DenseFormer(static), MRLA(dynamic-linear-ish) — see all prior outputs. **AttnRes Full/Block = dynamic × all-individual-earlier-layers × softmax** via one cheap query.

## Critical read (keep in mind)

Strengths: elegant unifying framing; ~0 extra params; the systems work is the real enabler; gains consistent (scaling laws + every benchmark) and mechanism visible (magnitude/gradient plots); reasoning/code gain most. Caveats: effect sizes small (~0.02–0.03 val loss); 1.25× is extrapolated (no CIs); Full-vs-Block "0.001" is noise; Block ≈ mHC (only Full clearly leads, at more memory); largest run is 3B-active/1.4T tokens so frontier persistence is assumed; hyperparams tuned for the baseline; no seeds/variance; L is tiny so the "softmax-vs-linear" framing may outrun the actual win (which largely comes from breaking fixed weights + RMSNorm magnitude control).

## Verdict

Low-risk, promising, systems-hardened drop-in upgrade to the residual stream; credible and consistent, but absolute gains small and the decisive comparisons (vs mHC, vs retuned baseline, at frontier scale) unsettled. Most transferable insight: **retaining distant layers beats many nearby ones** (sliding-window only reached 1.764).

## Notes for regeneration / serving

- The HTML must stay self-contained (no CDN/network). Keep inline `<style>` + one `<script>` block (theme toggle, scroll-spy, progress bar). Verify well-formedness with `python3 -c "from html.parser import HTMLParser; ..."` if you edit it; check balanced `div`/`section`/`svg`/`table` and that `<svg>` uses `viewBox`, `<marker>` defs present.
- To read locally: `python3 -m http.server 8137` in this repo, then open `http://127.0.0.1:8137/attnres-deep-dive.html`. (Prefer a non-blocking launch; avoid foregrounding `curl` against an empty host.)
- Always label the page a third-party summary and link the original: abs `https://arxiv.org/abs/2603.15031`, PDF `https://arxiv.org/pdf/2603.15031`, code `https://github.com/MoonshotAI/Attention-Residuals`.
