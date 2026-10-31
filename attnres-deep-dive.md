# Attention Residuals — Deep-Dive Explainer

> **Paper:** *Attention Residuals* — Kimi Team (Moonshot AI), arXiv **2603.15031**, cs.CL, submitted 16 Mar 2026.
> **Companion to:** `attnres-deep-dive.html` (the styled single-page version, now with a **§11 live demo**: a real tiny GPT that trains in-browser, Standard residual vs AttnRes, animated before/after). This is the readable Markdown twin.
> *Third-party summary; verify against the [original](https://arxiv.org/abs/2603.15031).*

---

## 00 · TL;DR

> Swap the transformer's **dumb, equal-weight additive residual stream** for a **cheap learned attention that decides, per layer, which earlier layers to listen to** — so deep networks stop drowning out their own layers and can selectively reuse early information — and engineer it to be a near-free drop-in.

| | |
|---|---|
| Residual-stream memory / pipeline comms | **O(L·d) → O(N·d)** via block summaries (N ≈ 8) |
| Compute advantage @ 5.6 PFLOP/s-days | **≈ 1.25×** (scaling-law fit) |
| Overhead | **< 2%** inference latency · **< 4%** train under pipeline |
| Extra parameters | **~0** — one vector + one RMSNorm per layer |

---

## 01 · The Problem: Residuals Are Dumb Sum-Machines

Standard PreNorm residual update — each layer just **adds** its output onto a running state with fixed weight 1:

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

Unroll it and a layer's input is the **unweighted sum of every earlier layer's output** (+ the embedding):

```
h_l = h_1 + Σ_{i=1}^{l-1} f_i(h_i)      # all weights fixed to 1
```

The residual does *two* jobs. The famous one is the identity path for gradients. The under-appreciated second: **it decides how information mixes across depth** — and here it uses fixed unit weights, with no way to emphasize or suppress any layer.

**Three failure modes from uniform accumulation**

1. **Growth** — hidden-state norm grows O(L); each layer's *relative* contribution shrinks ("PreNorm dilution"). Deep layers must output ever-larger values to be heard.
2. **Buried** — everything collapses into one blob, so a deep layer can't selectively retrieve an early output. Early info is irrecoverable; many layers can be pruned with little loss.
3. **Shared** — attention and MLP layers get the same aggregated state, though they'd ideally weight depth differently.

> **Why not just gate it?** Highway / DeepNorm / Hyper-Connections add learned or multi-stream mixing, but each stays *bound to the additive recurrence*: a layer only ever sees its immediate input `h_{l-1}`, a single compressed state that already collapsed history. They relieve compression without giving *selective access* to individual earlier layers.

---

## 02 · The Core Idea: Attention… Over Depth

The spark is a **duality**. RNNs over *time* collapse history into one state and only later got attention; residuals over *depth* do the exact same thing — compress all prior layers into `h_l`. So apply the same remedy: **let each layer attend across depth.**

```
h_l = Σ_{i=0}^{l-1}  α_{i→l} · v_i          # v_0 = h_1 (embedding); v_i = f_i(h_i) for i≥1
α_{i→l} = φ(q_l, k_i) / Σ_j φ(q_l, k_j)      # softmax over depth
φ(q, k) = exp( qᵀ · RMSNorm(k) )
```

- **Query = a bare learned vector.** `q_l = w_l ∈ R^d`, *decoupled from the forward pass* — this decoupling is why it stays cheap (§04).
- **Keys/values = the prior outputs themselves.** `k_i = v_i` — no extra projections.
- **RMSNorm on keys** stops a naturally-large layer output from hogging the softmax; ablation shows removing it hurts (more for the block variant).
- **Zero-init the query.** All `w_l = 0` ⇒ uniform weights ⇒ the model *starts* as an equal-weight average (≈ the old residual) and learns away from it. Keeps training stable.

---

## 03 · Full → Block: The Scalable Variant

**Full AttnRes** is free in a naive single-device loop — the outputs it attends over are already kept alive for backprop. At scale (activation recompute + pipeline parallelism) those outputs must stay alive for all later layers *and* ship across stages → **O(L·d)**.

**Block AttnRes**: partition L layers into N blocks of S = L/N. Within a block use ordinary additive residuals (a running partial sum); across blocks, attend **only over the N block summaries**:

```
b_n = Σ_{j ∈ B_n} f_j(h_j)      # one summary per block; embedding is always source 0 (b_0 = h_1)
```

| | |
|---|---|
| **Complexity** | Full: O(L²d) compute, **O(Ld)** mem/comms → Block: O(N²d), **O(Nd)** |
| **The knob N** | N = L ⇒ Full AttnRes; N = 1 ⇒ standard residual; **N ≈ 8** recovers most of the gain (~8 stored states/token) |

Each block's *first* layer sees prior summaries + embedding; each *subsequent* layer additionally attends to the growing intra-block partial sum `b_n^i` — preserving the local, near-immediate-predecessor path (the diagonal dominance the authors observe).

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = stack(blocks + [partial_block])   # [N+1, B, T, D]
    K = norm(V)                            # RMSNorm on keys
    logits = einsum('d, n b t d -> n b t', proj.weight, K)
    h      = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h                               # ← input for the next layer
```

AttnRes is applied **twice per transformer layer** — once feeding attention, once feeding the MLP, each with its own `w_l` + RMSNorm.

> **My take.** Depth is tiny (≤ ~64 sources), so AttnRes is practically a *learned weighted-sum over a handful of vectors*. The elegant linear→softmax framing is real, but much of the win likely comes from just **(a) breaking the fixed unit weights** and **(b) magnitude control via RMSNorm + softmax** — the paper's own input-independent-mixing ablation (1.749) and mHC baseline (1.747) already capture a lot of it.

---

## 04 · Making It (Almost) Free — The Systems Work

This is what earns the "drop-in" label; prior cross-layer-access methods were hard to scale.

- **Training · cross-stage caching.** Interleaved 1F1B repeats each stage V times (virtual stages). Cache blocks locally; ship only *incremental* blocks per transition. Peak per-transition cost **O(C) → O(P)**, a **V×** win, overlappable with compute.
- **Inference · two-phase + online softmax.** Block summaries act like a shared KV cache. *Phase 1*: batch all S queries in a block against the cache (one pass → outputs + log-sum-exp stats). *Phase 2*: fold in the sequential intra-block partial sum via online softmax. Amortizes S reads → 1.
- **Inference · sequence-sharded prefill.** A 128K sequence's block reps ≈ **15 GB**. Shard across TP devices (Phase 2 rides the all-reduce path) → ~**1.9 GB**/device; with 16K chunked prefill, < **0.3 GB**/device.

**I/O per token per layer** (residual mechanism only; typical L=128, N=8, S=16, m=4):

| Scheme | Total I/O (symbolic) | Typical |
|---|---:|---:|
| Standard residual | 3d | **3d** |
| mHC (m streams) | (8m+2)d + 2m² + 4m | 34d |
| AttnRes — Full | (S+N)d | 24d |
| **AttnRes — Block** | **(N/S + 5)d** | **5.5d** |

Reported net effect: negligible train overhead without pipeline parallelism, **< 4%** under it, **< 2%** inference latency.

---

## 05 · The Evidence

### Scaling laws (5 MoE sizes, 194M → 528M active)

Same exponent, lower intercept — a consistent shift across the compute range. Hyperparameters fixed for the *baseline* (deliberately conservative). Fitted `L = A·C^−α`:

| Variant | Fit | Val loss @ 5.6 PFLOP/s-days |
|---|---|---:|
| Baseline (PreNorm) | 1.891·C^−0.057 | 1.714 |
| **Block AttnRes (N=8)** | 1.870·C^−0.058 | **1.692** *(≈ 1.25× compute)* |
| Full AttnRes | 1.865·C^−0.057 | gap to Block → 0.001 at largest |

### Downstream (48B / 3B-active Kimi-Linear, 1.4T tokens)

Same recipe as the Kimi-Linear baseline (KDA+MLA 3:1 hybrid, MoE, Muon, WSD). **Matches or beats baseline on every task**; biggest jumps on multi-step reasoning & code.

| Area | Task | Baseline | AttnRes | Δ |
|---|---|---:|---:|---:|
| General | **GPQA-Diamond** | 36.9 | **44.4** | **+7.5** |
| General | MMLU | 73.5 | **74.6** | +1.1 |
| General | MMLU-Pro (hard) | 52.2 | 52.2 | 0.0 |
| General | BBH | 76.3 | **78.0** | +1.7 |
| General | ARC-Challenge | 64.6 | **65.7** | +1.1 |
| General | TriviaQA | 69.9 | **71.8** | +1.9 |
| Math/Code | **Minerva Math** | 53.5 | **57.1** | **+3.6** |
| Math/Code | **HumanEval** | 59.1 | **62.2** | **+3.1** |
| Math/Code | MBPP | 72.0 | **73.9** | +1.9 |
| Math/Code | GSM8K | 81.7 | **82.4** | +0.7 |
| Math/Code | CMath | 84.7 | **85.1** | +0.4 |
| Chinese | **C-Eval** | 79.6 | **82.5** | **+2.9** |
| Chinese | CMMLU | 82.0 | **82.9** | +0.9 |

### Ablations (16-head model, fixed compute — lower is better)

| Variant | Val loss | Read |
|---|---:|---|
| Baseline (PreNorm) | 1.766 | reference |
| DenseFormer (static cross-layer) | 1.767 | ✗ no gain → input-dependence matters |
| mHC (m streams) | 1.747 | strong competitor |
| **AttnRes · Full** | **1.737** | ★ best |
| **AttnRes · Block (S=4)** | **1.746** | near-free |
| └ input-independent mixing | 1.749 | learned query essential |
| └ sigmoid (not softmax) | 1.741 | competitive norm helps |
| └ no RMSNorm on keys | 1.750 | magnitude control matters |
| └ multihead (H=16) | 1.752 | depth-mix uniform across channels |
| Sliding-window (retain last 8) | 1.764 | **distant** access beats many near ones |
| input-dependent query | 1.731 | best, but needs d×d proj + sequential access |

Block size S degrades gracefully: S = 2,4,8 ≈ 1.746; S = 16,32 drift toward baseline. They fix **N ≈ 8** for infrastructure efficiency.

**Training dynamics:** Block AttnRes keeps output magnitude *bounded & periodic* (resets at block boundaries) and spreads gradient norm *more uniformly* across depth — the baseline overweights the earliest layers.

**Architecture sweep:** beats baseline in all 25 configs by 0.019–0.063; optimum shifts toward *deeper/narrower* (d_model/L_b 60 → 45). A diagnostic, not a deploy rec (deeper = slower).

---

## 06 · The Unifying View: Two Dualities

The intellectual spine: the *depth* axis and the *time* axis obey the same rules.

**① Sequence–depth duality**

| over time (RNN / TTT) | over depth (residual) |
|---|---|
| `W_t = W_{t-1} − η·∇ℓ(W_{t-1}; x_t)` | `h_l = h_{l-1} + f_{l-1}(h_{l-1})` |
| one "gradient step" per token; linear case = linear-attention `S_t = S_{t-1} + k_t v_tᵀ`; then Transformer → self-attention | `h_l` is "the state," each layer a "gradient step"; AttnRes → cross-layer attention |

Correspondences carry over: data-dependent gates (RetNet/GLA) ↔ Highway; delta-rule (DeltaNet/GLA) ↔ DDL; gated linear attention ↔ MRLA. All *refine* the recurrence; **AttnRes is the one that replaces it with attention.**

**② Residuals as structured matrices.** Write any variant as `h_l = Σ_{i<l} M_{i→l} v_i` for a depth-mixing matrix `M ∈ R^{L×L}`. The **semi-separable rank** of M classifies the family:

- **Standard** — all-ones lower-triangular; rank-1; fixed weights.
- **Highway** — cumulative gate products ⇒ rank-1, input-dependent; a "softmax-free stick-breaking attention."
- **(m)HC / DDL** — m parallel streams ⇒ m-semi-separable ⇒ **depth-wise *linear* attention** with matrix states.
- **AttnRes** — dense **rank-L** ⇒ **depth-wise *softmax* attention**; Block interpolates (rank ≈ N…N+S).

> **Bonus insight — attention sinks, on depth.** Because M is input-dependent, the matrix lens predicts *depth-wise attention sinks*: layers that attract high weight regardless of input — mirroring the sequence-side attention-sink phenomenon. The learned-weight heatmaps confirm strong diagonal (locality), persistent embedding weight (source 0), and off-diagonal "skip" concentrations. Pre-attention layers have broader receptive fields; pre-MLP layers are sharper/local.

---

## 07 · Where It Sits: Mechanism Map

Organized by what **weight** the mixing gets and what **sources** a layer can access, AttnRes is the only one simultaneously **dynamic** *and* reading **all individual earlier layers** through one cheap query.

| Class | Method | Weight | Sources layer l sees |
|---|---|---|---|
| Single-state | Residual | Fixed | h_{l-1} |
| Single-state | ReZero / LayerScale | Static | h_{l-1} |
| Single-state | Highway | Dynamic | h_{l-1} |
| Single-state | DeepNorm / KEEL | Fixed (+Norm) | h_{l-1} |
| Multi-state | SiameseNorm | Fixed | 2 streams |
| Multi-state | HC / mHC | Dynamic | m streams |
| Multi-state | DDL | Dynamic | d_v streams |
| Cross-layer | DenseNet / DenseFormer | Static | [h_1 … h_{l-1}] |
| Cross-layer | MRLA | Dynamic (linear-ish) | [h_1 … h_{l-1}] |
| **Cross-layer** | **AttnRes · Full** | **Dynamic** | **[h_1 … h_{l-1}]** |
| **Cross-layer** | **AttnRes · Block** | **Dynamic** | **[b_0 … b_{n-1}, b_n^j]** |

A one-dimensional "how smart is the residual stream" axis:
`Fixed → Static-learned → Input-dependent → Dynamic × all-layers × softmax (AttnRes)`

---

## 08 · Critical Read

**Strengths**

- Genuinely **unifying** framing (depth↔time duality, semi-separable rank) subsuming a dozen prior methods.
- **Minimal**: one vector + one RMSNorm per layer ⇒ params ≈ 0, drop-in.
- The **systems work** is the real substance — it separates "hard to scale" prior art from a usable component.
- Gains **consistent** (scaling laws *and* every benchmark); mechanism **visible** in magnitude/gradient plots.
- Reasoning/code gain most — a meaningful, interpretable pattern.

**Caveats**

- Effect sizes **small** (~0.02–0.03 val loss); "1.25×" from *extrapolated* fits (no CIs); Full-vs-Block "0.001" is noise-level.
- Win vs strongest rival **thin**: Block 1.746 ≈ mHC 1.747; only Full (1.737) clearly leads, and it costs more memory.
- **Scale not proven**: largest is 3B-active / 1.4T tokens; frontier persistence assumed.
- Hyperparameters tuned for the **baseline**; no seeds/variance; no stability ablation beyond a zero-init note.
- Grand framing may **outrun the regime** (L tiny); more tuning surface (N, block/stage alignment, per-sublayer queries).

---

## 09 · Bottom Line

> A well-motivated, elegantly framed, **systems-hardened** upgrade to the residual stream. It attacks a real PreNorm pathology with a minimal learnable depth-attention, and the block + pipeline optimizations earn the drop-in claim. The idea is credible and the reasoning gains encouraging — but the *absolute* improvements are small and the decisive comparison (vs mHC, vs a retuned baseline, at frontier scale) isn't fully settled.

Practical stance: treat it as a **low-risk, promising architectural tweak** whose value rises if the gains hold when you scale up — precisely the open question the paper leaves. The best ablation to internalize isn't even the attention: it's that **retaining distant layers beats many nearby ones** (sliding-window only reached 1.764) — reframing what "access" should optimize for.

> *"…completing for depth the same linear-to-softmax transition that proved transformative over sequences."* — the paper's thesis sentence.

---

## 10 · Glossary

| Term | Meaning |
|---|---|
| AttnRes | Attention Residuals — replace fixed residual accumulation with softmax attention over previous layer outputs. |
| Pseudo-query `w_l` | One learned vector per layer used as the attention query; decoupled from the forward pass, zero-initialized. |
| RMSNorm on keys | Normalizes layer outputs before scoring so large-magnitude layers don't dominate the softmax. |
| Block AttnRes | Group L layers into N blocks; attend over N block summaries (+ running partial sum) → O(Nd) cost. |
| PreNorm dilution | Pathology where O(L) hidden-state growth shrinks each layer's relative contribution. |
| Semi-separable rank | Complexity of the depth-mixing matrix M; classifies residual variants (rank-1 vs m vs L). |
| Online softmax | Milakov-style streaming softmax merge; lets the two-phase inference stay numerically identical. |
| 1F1B / virtual stages | Interleaved pipeline schedule; basis for the cross-stage block-caching trick. |

---

## 11 · Live on Real Hardware: GB10 GPU Training

The HTML page ships §11 as a **real GPU demo, not a simulation**: two identical tiny GPTs (~110k params, d=64, dff=128, 6 depth-ops) trained from scratch on the host's NVIDIA GB10 in PyTorch + CUDA, with genuine backprop — differing *only* in the depth-aggregation scheme, exactly as in the paper.

**The game, in one paragraph (no jargon).** Twenty words in a row; one of them happens to appear **twice**. The only question is: *the second time you see that word, what word came right after it the first time?* E.g. the word **year** appears and is followed by **area**; much later the list ends on **year** again, and the model must answer **area** — by *remembering what followed year the first time*. Nothing in that final **year** reveals the answer on its own; you have to **go back and look**. That look-back is the entire task — the smallest puzzle that forces a model to retrieve a fact from its own past — and the two racers differ **only** in how they carry information forward from past to present.

- **Task:** induction recall with k=20 — 20 pairwise-distinct tokens, then a copy of one of them in the final position; the model must predict that token's successor (the answer is `tok[j+1]`).
- **Data:** a seeded 60,000-example pool (seed 777); a fresh 144-row batch is drawn every step and served to **both arms in turn**, so the two models see the identical data stream; same init seed (1234), AdamW + warmup, one 3:00 window covers both (compute-fair interleaving).
- **Two dataset modes (toggle):** *token ints* — random ids 1–64: the controlled version; nothing can be memorized, the rule is exactly solvable, and the only difference between the arms is the wiring (a fixed/naturalistic corpus would confound the race with practice effects, vocabulary statistics, or simply unsolvable-but-realistic loss). *english words* — same rule and seeds, but the 20 context words are drawn **Zipf-weighted** (w ∝ 1/rank, without replacement) from the classic 64 commonest-English-nouns collection: the paper's own register, where rare-word sequences are the ones that should lean hardest on selective reuse of earlier (especially embedding) sources — exactly what the fixed residual sum is structurally blind to.

The minimum dataset both arms train on, verbatim rows of that pool (`[ ]` = first occurrence of the repeated token, `( )` = its copy in the final position, target to the right):

```text
1.  61 55 1 63 29 [27] 60 47 18 34 59 3 28 19 43 21 10 17 12 26 · (27) → predict 60
2.  13 3 24 54 44 26 19 39 64 37 [34] 45 52 57 61 47 46 14 29 38 · (34) → predict 45
3.  14 49 33 4 18 31 39 8 [32] 34 59 29 27 2 61 40 41 46 10 7 · (32) → predict 34
4.  31 54 45 [9] 22 49 37 34 59 15 33 62 41 16 4 5 21 63 18 2 · (9) → predict 22
5.  1 45 17 [35] 30 50 60 56 6 32 11 40 33 38 39 55 53 26 20 37 · (35) → predict 30
6.  19 5 56 37 47 17 44 36 12 25 40 55 29 9 30 54 14 46 24 43 [46] · (46) → predict 24
```

```text
1.  house world way time company people [year] area program story health system member night government woman child place week day · (year) → predict area
2.  child year hand area time [head] book work system place life day people world side right way woman house game · (head) → predict book
3.  job hand place life line time case day year government program world story work home member [part] car man week · (part) → predict car
4.  people life way month [system] part time home night year number job kind right book city point fact story place · (system) → predict part
5.  time man day month way story people issue child place book system year night mother house health [fact] friend home · (fact) → predict friend
6.  time child part night world year case home law way right man [government] week house company people health study day · (government) → predict week
```

What to watch: the loss/accuracy race (both arms learn the retrieval rule live), the per-depth `‖h_l‖` bars (std grows with depth — the O(L) PreNorm dilution — while AttnRes stays bounded), per-op gradient norms, and the learned `α` heatmap forming during real training. The page's dataset toggle renders both modes interactively (word rows are the same ids mapped through the 64-word collection); values are baked into the page as fallback so the exhibit survives the GPU server being offline.

**Two α views — the trend, and the real weights.** The first heatmap is batch- and token-averaged: one number per (aggregation point, source). The second is un-averaged: the server runs the trained AttnRes on the six verbatim example rows and reports the **final** aggregation's softmax per token — one column per actual token, seven stacked source segments (embedding at the bottom, then L1…L6), hovering a column giving the exact per-source weights and, in word mode, the word. On a ~700-step run the pattern is legible and specific: the final (repeated) token loads ~0.5–0.7 of its weight on a single early layer (L2 here) and only ~0.01–0.02 on the embedding, while each token's first occurrence draws visibly more on the embedding (~0.04–0.21) — selective, position-specific reuse that a fixed residual sum cannot represent. The panel is rendered ~2× taller than before (larger strips and type) so the per-token strips and their word labels read at a glance; on the latest 180 s / 7,067-step run the routing is sharper still — the final (repeated) token puts ~0.9972 of its weight on a single layer (L3), with only ~0.0002 on the embedding.

**The matmul, in action — the real `QKᵀ → softmax → ·V`.** A third live panel opens the hood on the attention that *does* the looking-back. For one example it takes a real trained head and replays it, step by step or on autoplay: the final token's **query** `q` dotted against every **key** `k` to form the raw score `q·k/√d` (one cell of the matrix `QKᵀ`); the causal mask; the **softmax** turning the score row into probabilities that sum to 1; and the weighted sum of **value** vectors (`Σ wⱼ·vⱼ`) that flows onward. Every number is the trained model's own, and the panel prints the literal multiply for the top cell (`q₀·k₀ + q₁·k₁ + … ÷ √d = score`) so the dot product visibly becomes the weight. The server scans **all** attention layers and heads and returns whichever best implements the look-back (falling back to the most decisive head early in training). On the latest run the winner is unambiguous: **L3 head 0** puts **0.9998** of its final-position attention on the single key at the answer position (token-ints sample: position 6 = the answer `60`) — i.e. a clean, learned **look-back head**. A baked real example ships as fallback, so the exhibit still works with the GPU server offline.

**Play it yourself, watch the head form, and an honest scoreboard.** Three more live widgets share the one sample, so nothing is scattered. (1) **Beat the model** — pick a deck (*english words / token ints / melody ♪ / letters / trap-with-no-repeat*), see a fresh 21-token tape, click the token you think comes next, then reveal the model's **top-5** next tokens as bars, a **copy-beam** arc from the repeated token back to the position the head actually reads, and a running you-vs-model score. The deck is only a face on the ids — the model never sees words or notes, just numbers 1–64 — and the server answers from the **persisted trained weights** (`.gpu_last_model.pt`) via `/api/gpu/play`, so the game works even between runs. A **released, verified checkpoint** of the same architecture — [huggingface.co/xcpatr922JNP/attnres-tinygpt-induction](https://huggingface.co/xcpatr922JNP/attnres-tinygpt-induction) (`load_model("xcpatr922JNP/attnres-tinygpt-induction")`, MIT, strict-loads 47/47) — is a *different training run* from the live one, and answers all six example rows correctly. (A *weekdays* deck was cut: seven symbols can't make the 20-distinct tape the task needs, and short tapes throw the look-back head off.) (2) **Watch the copy head form** — every 25 training steps the server snapshots the look-back head's attention row; a heatmap (training-checkpoint × position) with a scrubber/autoplay shows a real head **crystallize**: on the latest run the spike is ~0.10 at step 25 and 1.00 on the answer token by the end (the traced head is re-picked at each checkpoint as the *best* look-back head, so its layer/head index can move early before it settles — the behaviour, not a fixed head, is the point). (3) **The race** — a scoreboard marks the first step each arm reaches 100% train accuracy, with curve toggles. **Honest result:** on this tiny 6-op toy the plain residual usually **memorizes the induction task first** (~step 2,600 vs AttnRes's ~3,500 on the latest run), and both finish at 100% and ~0 loss. AttnRes's win is what the panels above already show — bounded `‖h‖` with depth and even per-op gradients — not a sprint win, and the caption says so plainly.

**Implementation & serving.** `gpu_train.py` (task, models, training loop) + `gpu_demo_server.py` (loopback-only `127.0.0.1:8477`, SSE stream) sit beside the static origin of the public site; a tunnel ingress rule routes `/api/gpu*` to the GPU server and everything else to the static origin (`/api/gpu/status|snapshot|stream|data|play`, plus `start|stop`). Hardening: `start`/`stop` require an allow-listed `Referer`/`Origin` (403 otherwise), a global 20 s cooldown, a per-client-IP minimum interval (30 s) and hourly quota (12), and the SSE fan-out is capped (32 concurrent viewers). The in-flight guard runs **before** the rate limiter, so pressing start again while a run is live is rejected (`run already in progress`) **without** consuming the hourly quota; `/api/gpu/status` reports `cooldown_left`/`ip_next_in`/`ip_runs_left` and the page disables the button (with a countdown) while running or cooling down. Duration is clamped (20–300 s) and no user input ever reaches code execution. The snapshot carries a per-token `att_tok` block (the size of one small forward pass), the run's `ds`, and an incrementing `run_id`; the last completed snapshot is persisted to `.gpu_last_snap.json` and restored on boot, so a server restart keeps the last real result on screen. The server sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and retries a run once after a `cuda.empty_cache()` on CUDA OOM, since the host's unified memory is shared with a large LLM worker.

**A note on variance.** Every start trains a *fresh* model from scratch, so the exact head/layer and its weights differ run to run — the look-back head is usually in the middle layers but its index and width move. What is stable (and the actual point) is the *behaviour*: one head reliably puts ~0.99–1.00 of its final-position attention on the answer token. Each panel is labelled with its run (`run N`, or `reference run` for the shipped fallback — presently the run-32 snapshot above) so figures are never confused across runs. Each snapshot also ships a downsampled history of the whole run, so the loss/accuracy chart shows the full curve even when you open the page after a run has already finished, and an **auto re-run** toggle can chain runs back-to-back (waiting out the short cooldown). A private owner key (`?key=…`) lifts the cooldown and hourly cap for the host's own use.

**Latest run (run 32, GB10).** 180 s, 7,067 steps from scratch: both arms reach **100%** training accuracy with loss ≈1–3×10⁻⁴. The AttnRes arm's final `‖h‖` stays bounded at **18.07** while the plain-sum arm climbs to **32.85** — the O(L) dilution the paper names, measured on real hardware rather than asserted. The trained depth-α concentrates on **L3 (99.7%** of the final read; ~0.02% on the embedding); the sharpest look-back head is **L3 head 0**, putting **0.9998** of the final token's attention on the single answer position; and that head crystallizes from ~0.10 at step 25 to 1.00 by the end. The refreshed `.gpu_last_model.pt` — 6/6 on the six verbatim example rows, 110,593 params — is what `/api/gpu/play`, `/api/gpu/anatomy` and the §11 widgets now serve, so the numbers above and the live panels describe one and the same model.

---

## 12 · Network Programmer View

A framing from the ops side of the house: if you've configured multicast routing, the machinery inside attention will feel eerily familiar. Attention is **content-addressed, soft multicast** — a pub/sub bus where the subscription match is a *learned similarity function* instead of an exact group address.

> Programming-wise, attention is a pub/sub bus where the subscription match is similarity instead of equality — and delivery is **replicate-then-reduce**, not replicate-only.

### The concept mapping

| Multicast | Attention |
|---|---|
| Receiver joins a group (IGMP/PIM join) | Query — "what I'm interested in" |
| Group address / (S,G) label | Key — "what I advertise" |
| Payload | Value |
| Join matching a group | Dot product q·k |
| Replication to members | Weighted fan-out of values |
| RPF check / ACL | Causal mask |
| Multiple VRFs / group spaces | Multi-head attention |
| mroute state | KV cache |
| Dense mode (flood and prune) | Full O(n²) attention |
| Underlay unicast path | **Residual connection** |

### The key difference: hard vs soft membership

In multicast, membership is a **hard 0/1 matrix M**, set by signaling, and each receiver gets separate copies of what it joined. In attention, membership is a **soft, row-normalized matrix** `A = softmax(QKᵀ/√d)`, computed from the data itself, and each receiver sums what it receives:

```python
n, d = 5, 8
Q, K, V = (np.random.randn(n, d) for _ in range(3))
# Multicast: static, hard membership, no combining
M = (np.random.rand(n, n) > 0.5).astype(float)      # who joined whose group
mcast_rx = [V[M[i] == 1] for i in range(n)]          # each receiver gets a list of copies
# Attention: dynamic, soft membership, replication + reduction
S = Q @ K.T / np.sqrt(d)
A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
attn_out = A @ V                                      # weighted sum per receiver
```

So attention is not *pure* multicast. It is multicast **plus in-network aggregation** — closer to a dynamic all-to-all with reduction. A pure multicast tree never transforms or combines payloads; attention does both.

### The same interface, two implementations

```python
class HardBus:  # Multicast: exact match on group, deliver copies.
    def __init__(self): self.members = {}            # group -> [receiver ids]
    def join(self, rx, group): self.members.setdefault(group, []).append(rx)
    def send(self, group, payload):
        return {rx: payload for rx in self.members.get(group, [])}

class SoftBus:  # Attention: similarity match, deliver a weighted blend.
    def __init__(self): self.keys, self.vals = [], []
    def publish(self, key, value): self.keys.append(key); self.vals.append(value)
    def receive(self, query, temp=1.0):
        K, V = np.array(self.keys), np.array(self.vals)
        s = K @ query / temp
        w = np.exp(s - s.max()); w /= w.sum()         # softmax = soft join
        return w @ V                                   # ONE merged delivery
```

| Multicast code | Attention code |
|---|---|
| `dict[group] -> [rx]` | `softmax(Q @ K.T)` |
| `IP_ADD_MEMBERSHIP` | computing a query vector |
| `sendto(group, data)` | `publish(key, value)` |
| `recvfrom()` returns N separate datagrams | `A @ V` returns one blended vector |
| `if group in table` (boolean) | float weight in [0, 1] |
| Membership is state you mutate | Membership is recomputed every forward pass |

### The temperature knob: exact-match ↔ flooding

Low temperature: softmax → argmax, a hash lookup — the multicast dict, or unicast forwarding. High temperature: weights go uniform — flood-to-all with averaging. Temperature is a knob sliding between exact-match forwarding and flooding.

### Where the analogy breaks, in code

- **No reduction in multicast.** `recvfrom()` gives you N datagrams to handle; attention sums them *before you see anything*.
- **Differentiable.** The match is smooth, so gradients flow back and the system *learns what to subscribe to*.
- **Batched.** Attention does all joins and sends in one matmul; a socket-level model would loop per receiver.

Two honest push-backs: (1) multicast is sender-push with signaling (IGMP joins, PIM hellos, prune timers — a protocol state machine), while attention is receiver-pull with membership *recomputed from data every pass* — the closer cousin is **content-based pub/sub** with a learned topic vector. (2) The reduction is the deepest disanalogy: classic multicast routers never combine payloads, which is why AI fabrics needed a separate capability — in-network reduction, as in InfiniBand SHARP (which predates attention and came from collectives; credit where due).

### Where it connects physically

- **Collectives:** sharded attention sees all-gather / reduce-scatter / all-reduce — the physical multicast-and-aggregate patterns.
- **In-network compute:** SHARP on InfiniBand and multicast on NVSwitch exist because replicate-and-reduce is so common.
- **MoE routing:** a router picks top-k experts per token → data-dependent, bursty all-to-all; imbalance resembles ECMP hash-polarization.
- **Positional encoding:** plays the role of sequence numbers or TTL.

**Tie-back:** the row "underlay unicast path = residual connection" is the hook — standard residuals give every layer one fixed, equal-weight delivery path (a static tree). **AttnRes applies content-addressed delivery to the depth axis itself** — the same move the sequence axis made in 2017. The residual stream is the multicast-like substrate, and AttnRes rewrites its underlay, making depth content-addressed exactly as attention did for tokens.

---

## 13 · The Medium Read: The Paper, and What We Rebuilt

Every large language model you have ever used is built from dozens of processing layers stacked on top of each other, and for nearly a decade the wiring between those layers has not fundamentally changed: **each layer's output is simply added to a running total, with a fixed weight of exactly 1.** A March 2026 paper from Moonshot AI's Kimi team asks — *what if that wiring could learn?* — and the answer appears to make every model it touches slightly better. This section is the plain-language version of everything above, including our own attempt to reproduce it.

### 1 · The problem: a meeting where everyone talks at once

Imagine a company where every decision passes through fifty review committees in sequence. Each committee writes a memo, and every memo is added — verbatim, with equal weight — to a shared binder that all later committees must read. By committee forty, the binder is enormous. New committees have to write *louder and louder* memos just to be noticed over the accumulated bulk, and a crucial insight from committee three is buried under everything that came after.

That binder is the **residual stream**, and the paper documents what the meeting analogy suggests: the stream's magnitude grows steadily with depth (**PreNorm dilution**), early information gets buried and cannot be selectively retrieved, and you can delete large chunks of deep models with almost no quality loss — a symptom of how redundant the accumulated binder has become.

**The fix, in one picture.** Standard: every output, equal weight (+1), hidden state grows with depth. AttnRes: each layer scores all earlier outputs and reads a learned, weighted blend (weights recomputed for every token, summing to 1).

### 2 · The fix: teach the wiring to listen selectively

**AttnRes** replaces the fixed add with a learned choice. Each layer gets a tiny learned "query" vector, scores every earlier layer's output, the scores become percentages (sum to 100%), and the layer reads a weighted blend instead of the undifferentiated pile. Every committee gets a one-line index of everything said so far, and the freedom to tune into the two or three memos that actually matter.

The elegance is in the budget: the mechanism costs **one learned vector per layer** — a rounding error against billions of parameters — and it is a drop-in replacement. The authors frame it as completing, for depth, the same upgrade attention gave to sequence processing back in 2017.

> Standard residuals give every layer one fixed, equal-weight delivery path. AttnRes lets the wiring decide where to look.

### 3 · Does it work? What the paper measured

- **Scaling laws.** Across five model sizes the improved wiring shifts the loss curve down consistently — worth roughly **1.25× the compute** at the tested scale.
- **A production-scale trial.** 48B params (3B active), 1.4T tokens: **matched or beat the standard version on every benchmark**, with the largest gains in multi-step reasoning and code (GPQA-Diamond 36.9 → 44.4; math +3.6; code +3.1). Single-run numbers, no seed variance; the "matched or beat" line includes one dead tie on MMLU-Pro.
- **The mechanism, visible.** Hidden-state growth becomes bounded and gradients flow more evenly across depth — the disease and the cure, both observable.

The engineering is half the story: the block-based variant keeps overhead at **under 2% inference / under 4% training**, which makes this a deployment candidate rather than a paper curiosity.

### 4 · What we did: rebuilt it, at toy scale

We rebuilt the comparison at a scale small enough to watch: two identical models on a real NVIDIA GPU in PyTorch, in a three-minute session you can trigger yourself (§11) — identical data, identical seeds, differing *only* in the wiring under test. The result matches the paper: under standard residuals the hidden-state magnitude climbs steadily with depth (≈ 11 → 27.5 and still climbing in our run); under AttnRes it stays bounded (≈ 18.4 at the last layer, with block resets). Both variants learn the task.

**What a toy scale can and cannot show:** it demonstrates the *mechanism* — the pathology exists, and the fix changes it — but not the *quality* claim: both models still solve this small task. The benchmark gains in point 3 require the paper's full-scale runs, or your own.

### 5 · The skeptic's corner

The improvements, while consistent, are modest in absolute terms; the headline compute advantage comes from extrapolating fitted curves; the largest validated model is far from frontier scale; the closest competing approach (mHC) ties in head-to-head ablations; no seed variance is reported. What balances the scales is the asymmetry of the bet: the change costs approximately nothing for a consistent, mechanism-explained improvement. Low-risk architectural upgrades with plausible upside are rare; when one appears, the rational move is to test it, not wait for certainty.

### 6 · Takeaways for decision-makers

- **Near-free upgrades to core infrastructure deserve attention.** This touches the most universal component in modern AI — the residual connection — and costs essentially nothing to adopt.
- **The deepest insight generalizes beyond AI.** Giving later stages *selective access to earlier ones* outperforms handing them a single blended summary — in data pipelines, caching layers, and organizational memory too.
- **Reproduce the mechanism — it is cheap; the claim is not.** A toy rebuild makes the pathology and its fix observable in minutes, but cannot validate the quality gain. Cheap reproduction separates "the mechanism is real" from "the benefit is real", and only the first is a three-minute question.

*About this page:* a third-party explanatory summary of *Attention Residuals* (Kimi Team / Moonshot AI, arXiv 2603.15031v1) — not the paper, not affiliated with the authors; verify claims against the original. Tables reproduce the paper's numbers; "My take" / "Critical Read" boxes are interpretation.

---

*Links:* [arXiv abstract](https://arxiv.org/abs/2603.15031) · [PDF](https://arxiv.org/pdf/2603.15031) · [code](https://github.com/MoonshotAI/Attention-Residuals) · [weights (Hugging Face)](https://huggingface.co/xcpatr922JNP/attnres-tinygpt-induction)
