# Attention Residuals (AttnRes) — deep-dive explainer

A third-party explainer of **"Attention Residuals"** (Kimi Team / Moonshot AI,
[arXiv 2603.15031](https://arxiv.org/abs/2603.15031), cs.CL, 16 Mar 2026), plus a live
GPU demo that re-runs the paper's core claim on real hardware.

> **The idea in one line:** replace the transformer's fixed, equal-weight additive residual
> stream with a cheap learned softmax attention *over depth* — each layer decides which earlier
> layers to listen to — engineered as a near-free drop-in (~0 extra params, <2% inference latency).

This is a summary and reconstruction, not the paper. Verify against the original.

## What's here

| File | What it is |
|---|---|
| `attnres-deep-dive.md` | The readable explainer: problem → method → block-scalable variant → systems work → evidence → mechanism map → critical read → glossary (§00–§10), live-demo writeup (§11), network-programmer view (§12), plain-language version (§13). |
| `attnres-deep-dive.html` | The same content as a styled single-page site, self-contained (inline CSS, no CDN). **§11 is a live demo**, not a screenshot. |
| `attnres-deep-dive-implementation.md` | Engineering companion: how §11 is wired end-to-end — the two training arms, the SSE server, the page's panels, data flow, known issues, operational commands. Read this before touching the demo. |
| `.opencode/skills/` | Agent skills: `attention-residuals` (paper knowledge + how to regenerate the explainer) and `attention-residuals-gpu-demo` (operating knowledge for the demo). |
| `demo-verify.png` | Captured verification shot of the running demo. |

## The §11 demo, in brief

Two identical tiny GPTs (~110k params, d=64, 6 depth-ops) — standard PreNorm residual vs AttnRes —
train interleaved on an NVIDIA GB10 (PyTorch/CUDA, real backprop, one 3-minute window) on induction
recall: fresh 144-row batches from a seeded 60k pool, both arms sharing the same init seed. Panels:
loss race, `‖h‖` and gradient bars, α heatmaps (batch-averaged and per-token), a step-by-step replay
of a real `QKᵀ → mask → softmax → ·V` matmul, a "beat the model" next-token game, and a copy-head
crystalliser. Baked fallbacks keep every panel rendered with the server offline.

**Honest result, stated plainly in §11:** on a 6-op toy the plain residual usually learns induction
*first*. AttnRes's win here is bounded `‖h‖` with depth and more even gradients — not a sprint.

The demo's backend (`gpu_train.py`, `gpu_demo_server.py`) lives beside the public static site in
`~/projects/ai-research-pub/`, not in this repo; editing it requires restarting
`gpu_demo_server.py`. See the implementation doc for routes, rate limits, and the publish flow.

## Reading it

Open `attnres-deep-dive.html` in a browser — no build step, no network needed to read it, and every
demo panel still renders offline from baked fallbacks. `attnres-deep-dive.md` is the same text for a
terminal or code host. §11's live panels need the GPU backend described above; with no server they
fall back to the last captured numbers.

## License

CC BY-NC-ND 4.0 (`LICENSE`), matching the headers already declared in `.opencode/skills/`. Attribution
to this repository; no commercial use; no derivatives. The explainer summarises a third party's paper —
check that paper's own terms before reusing its figures or numbers.
