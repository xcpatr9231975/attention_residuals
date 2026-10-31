# AttnRes deep-dive — how the live GPU demo actually works

Engineering companion to `attnres-deep-dive.{html,md}`. This documents the **implementation**: how §11
("Live on Real Hardware") is wired end-to-end — the two training arms, the SSE server, and the
single self-contained HTML page that drives it (loss race, magnitude/gradient bars, α heatmaps,
the attention matmul replay, the "beat the model" game, the copy-head crystalliser, the honest race
scoreboard). It is *not* the paper explainer (that is `attnres-deep-dive.md`); this is the "how does
the thing under the hood work" twin, for whoever maintains it next.

---

## 1. The moving parts

| File | Role |
|------|------|
| `ai-research-pub/gpu_train.py` | the task, both models, `run_demo`, and every analysis probe (`attn_vis`, `token_alphas`, `analyze`, `examples`, `replay_payload`). **Single source of truth** for the constants. |
| `ai-research-pub/gpu_demo_server.py` | loopback-only HTTP `127.0.0.1:8477`; drives `run_demo` on a worker thread, fans out snapshots over SSE, exposes the JSON routes, persists the last run + weights. |
| `ai-research-pub/origin_server.py` | static origin `127.0.0.1:8097` for the public site. |
| `~/.cloudflared/config.research.yml` | tunnel ingress: `/api/gpu*` → `:8477`, everything else → `:8097`. |
| `attention_residuals/attnres-deep-dive.html` | the page. One inline `<style>` + two `<script>` blocks; §11 is the second script's IIFE. Self-contained (no CDN); baked fallbacks let every panel render with the server offline. |
| `attention_residuals/attnres-deep-dive.md` | the readable paper twin, kept in sync through §13. |
| `.gpu_last_snap.json` / `.gpu_last_model.pt` / `.gpu_owner_key` | persisted last full snapshot; the last trained AttnRes weights (for inference); the private owner bypass key. |

Publish flow: edit in `attention_residuals/`, `cp` the `.html`+`.md` to `ai-research-pub/attention/`,
bump the `?v=N` cache-buster in the shared link. Backend edits (`gpu_train.py`,
`gpu_demo_server.py`) require a **server restart** — a stale process silently ignores new params.

---

## 2. The experiment (`gpu_train.py`)

**Constants (line 19-26).** `K=20` distinct prefix tokens, `VOCAB=64` (embedding table `VOCAB+1=65`),
`POOL_N=60_000`, `BATCH=144`, `SEED=777` (pool), `INIT_SEED=1234` (both arms), `DURATION=180s`,
`CADENCE=25`. Sequence length is always `K+1 = 21`.

**Task — vectorised induction recall.** Build `k` pairwise-distinct tokens `t[0..k-1]`, pick
`j ∈ [0, k-2]`, append a copy `t[j]` at the final position; the label is `t[j+1]` (predict the
*successor of the first occurrence* of the repeated token). Nothing in the final token predicts its
own successor — the model must look back. Fresh batch every step from the seeded pool, so neither arm
can memorise a fixed set.

**Two dataset modes**, same geometry/seeds:
- `tok` — ids `1..64` uniform. The controlled race.
- `word` — the 20 context words drawn **without replacement, Zipf-weighted** (`w ∝ 1/rank`) from the
  64-noun `WORDS` list. The paper's register; labels are applied client-side only.

**The two arms** (`TinyGPT`, line 371) are identical except `scheme`:
- `std` — `h = Σ parts` (PreNorm residual sum; `_agg` returns the plain sum).
- `attn` — `DepthAgg` (line 313): `logits = einsum(w, RMSNorm(stack(parts)))`, `α = softmax(logits)`,
  `h = Σ α·parts`, with **zero-init** `w` so it starts as a uniform average (≈ the old residual).
Ops = `("attn","mlp")*3` → attention at op-index 0,2,4 (L1,L2,L3). `d=64, dff=128, vocab=65`.
The forward (line 390) records per-op `hNorm` and per-op mean `α` into `last_stats`, and keeps
`last_outs` (used by `grad_norms`). `CausalSelfAttn` optionally stashes `q/k/v/scores/weights`
when `.capture` is set (used by `attn_vis`/`analyze`).

**`run_demo` (line 425)** trains them **interleaved**: each step, for each arm it draws its *own*
fresh batch from the shared seeded sampler, `loss.backward()`, AdamW with a 150-step warmup.
Note the fairness guarantee precisely: identical distribution, identical init, identical schedule —
but *not* literally the same batch (two `sampler.next()` draws per step). Every `CADENCE` steps it
calls `snapshot()` and `emit`s; at the end it emits `done:True` and stashes `run_demo.att/.std` for
later inference.

**`snapshot()` (line 443)** assembles the payload: `loss/acc`, `hNorm_std/attn`,
`gradNorm_std/attn`, mean `att`, `att_tok` (per-token α), `attn_vis`, plus the downsampled `hist`
(`_hist_payload`, ≤160 pts) and `replay` (`replay_payload`, ≤40 frames). ⚠ see §6 — the read order
here is load-bearing and currently wrong for the AttnRes arm.

### The analysis probes

- **`attn_vis(model, rows)`** (line 200) — one `no_grad` forward with capture on; scans every
  attention head, ranks by `max(W[h,end,j], W[h,end,j+1])`; picks a genuine look-back head
  (`picked:"induction"` when ≥0.15) else the lowest-entropy ("decisive") head. Emits `q,k,v,scores,
  weights, look_key, look_w, answer_key, layer, head, dh(=16), picked, tokens, words, j, target`.
- **`token_alphas`** (line 139) — final-aggregation α per token, un-averaged (`α[source][token]`,
  sources `emb,L1..L6`).
- **`analyze(model, tokens, j)`** (line 261) — one forward on **any `T≤21`**: top-5 softmax, entropy,
  and the best look-back head's final-position attention row (the "copy beam"). Powers `/api/gpu/play`.
- **`examples`** (line 101) — the verbatim rows the page displays, from the same seeded pool.

---

## 3. The server (`gpu_demo_server.py`)

ThreadingHTTPServer on `127.0.0.1:8477`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
**before** importing torch (shared-memory host), and `run_thread` retries once after
`cuda.empty_cache()` on an OOM.

**Boot:** `_load_last_snap()` restores the last full snapshot; `load_model()` rebuilds an `attn`
model (`build_model`) and loads `.gpu_last_model.pt` into the global `MODEL`.

**Routes (GET):**
- `/api/gpu/status` — `running,done,step,t,error,ds,run_id,remaining,cooldown_left,ip_next_in,ip_runs_left,owner`.
- `/api/gpu/snapshot` — the latest snapshot (or `{step:0}`).
- `/api/gpu/stream` — SSE; replays the current snapshot then each new one, `: keepalive` every 15s,
  breaks on `done`, `finally` removes the subscriber. Cap `MAX_SUBS=32` → 503.
- `/api/gpu/data?n&mode` — the examples payload.
- `/api/gpu/start?dur&ds&key` — **in-flight guard runs BEFORE the rate limiter** (so stray clicks don't
  burn the quota); spawns `run_thread`.
- `/api/gpu/stop` — sets the STOP event.
- `/api/gpu/play?mode&j&seed&seq&norepeat&v` — the game. Builds a tape (via `_build_sequence` or a
  supplied `seq`), runs `analyze` on the **live** `run_demo.att` if present else the persisted `MODEL`
  (both under `INFER_LOCK`), returns `tokens, j, target_id, correct, pred_id, topk, entropy, attn`.

**Hardening.** `start`/`stop` require an allow-listed `Referer`/`Origin` (`GPU_DEMO_HOSTS`, else 403);
global `START_COOLDOWN=20s`, per-IP `IP_MIN_INTERVAL=30s` + `IP_MAX_RUNS=12`/hour (429); `dur` clamped
20–300s; client IP from `Cf-Connecting-Ip`/`X-Forwarded-For`. Owner = allow-listed IP
(`GPU_DEMO_OWNER_IPS` ∪ loopback) **or** constant-time `hmac.compare_digest(key, OWNER_KEY)`; owners
skip cooldown + quota. **`curl`-testing start/stop needs `-H "Origin: https://<public-host>"`.**

**Persistence.** `broadcast()` writes `.gpu_last_snap.json` (atomic `os.replace`) only for a *full* run
(`t ≥ 0.85*duration`); `run_thread` calls `save_model()` after a successful run.

---

## 4. The page (`attnres-deep-dive.html`, §11 script)

A single IIFE. `el{}` caches ~40 ids; `if (!el.gpuLoss) return;` bails if absent. State lives in two
objects:
- `S` — charts/run state: `hist{std,attn}`, `acc{std,attn}`, `hStd,hAtt,gStd,gAtt`, `att`, `attTok`,
  `attnVis`, `step,t,running,done`, `source` (EventSource), `poll`, `cooldown,runsLeft,owner`,
  `auto,autoPending,histRun`, plus `showStd/showAttn/replay`.
- `P` — game state: `mode,data,guess,you,bot,busy,rpidx,rpTimer`.

**Baked fallbacks** (`DATA_FB`, `DATA_FB_WORD`, `ATTN_FB`, `ATTK_FB`, `SNAP_FB`) are real values from
the last good run, applied via `apply(SNAP_FB)` at boot so every panel is alive before the server answers.

**Data ingress.** `apply(snap)` (line 1512) is the single sink: it writes `S.*`, rebuilds the whole
curve from `snap.hist` (or appends a point), adopts `snap.replay` (and sets the scrubber max), resets
the curve on a new `run_id`, recomposes the status line (adding the `‖h‖ last: std vs attn` tail when
done), and calls `draw()`. Transport: `listen()` opens `EventSource('/api/gpu/stream')`, and on any SSE
error falls back to `setInterval(poll, 800)` hitting `/api/gpu/snapshot`. `refreshStatus()` (4s) pulls
`/api/gpu/status`, tracks the cooldown countdown, and auto-opens the stream when a run appears.

**Run control.** `startRun()` posts `start?ds=…&key=…`; the auto-rerun toggle sets `S.auto` and a 1s
tick chains runs (waits the cooldown, respects the quota unless owner). `stop` disarms auto and posts
stop. The owner key comes from `?key=` in the page URL and is forwarded to `status`/`start`.

**Drawing** — all canvas, `frame2d(id)`: `drawLoss` (two loss curves + two dashed accuracy curves,
respects `showStd/showAttn`), `drawBars` (‖h‖ and ‖∂loss/∂v‖ side by side), `drawAttn` (mean-α
heatmap), `drawAttTok` (per-token stacked α + tooltip), `drawAttnMul` (the QKᵀ→mask→softmax→·V replay,
5 steps, click a row to change the query), `drawReplay` (the copy-head crystalliser: rows = checkpoints,
cols = positions, gold tick = first occurrence, scrubber + play), `updateRace` (first arm to acc ≥0.99).

**The game.** `PLAYM` maps deck→`{v,label,norepeat}`: `word`(64), `int`(64), `note`(21 → 7 notes ×
3 octaves), `letter`(26), `trap`(64, `norepeat`). The deck is a *face on ids* — the model only ever sees
`1..V`. `playNew()` fetches `/api/gpu/play`, `playRender()` paints 21 chips + candidate buttons,
`playGuess()` records your pick, `playReveal()` draws the top-5 bars + verdict + the copy-beam
(`drawPlayBeam`, an SVG path) + target highlight. ⚠ `playReveal` must compute its own
`last = d.tokens.length-1` (it was `playRender`-scoped) or the beam/underline silently no-op.
`playHear()` renders the melody deck with WebAudio.

---

## 5. End-to-end data flow

```
      run_thread ── run_demo(duration, emit, stop, ds)
                       │  step loop: for arm in [std,attn]: draw batch, fwd, bwd, step
                       │  every CADENCE: snapshot() ──► emit(snap)
                       │                                 ├─ _hist_payload (≤160)   ── loss/acc curve
                       │                                 ├─ replay_payload (≤40)   ── copy-head frames
                       │                                 ├─ attn_vis + token_alphas ── matmul + per-token α
                       │                                 └─ hNorm/gradNorm (both arms)
                       ▼
      broadcast(snap) ──► STATE["snap"]  ┬─► per-subscriber Queue ──► /api/gpu/stream (SSE)
                       └─ full run? ─► .gpu_last_snap.json            └─ /api/gpu/snapshot (poll fallback)
                       └─ end of run ─► save_model() ─► .gpu_last_model.pt ─► /api/gpu/play (analyze)

      PAGE: EventSource ──► apply(snap) ──► S.* ──► draw() {loss, bars, α, attok, matmul, replay, race}
            game: [data-pmode] ─► playNew() ─► fetch /api/gpu/play ─► playRender/Guess/Reveal
```

---

## 6. Known issues (item 1 was a real bug — now FIXED & verified; 2–4 are by design)

1. **`gradNorm_attn` used to read `[0,0,0,0,0,0]` — the AttnRes gradient panel was fake (FIXED).**
   `snapshot()` called `attn_vis(att, ex_rows)` at its **top**; that `no_grad` forward reassigns
   `att.last_outs` to gradless tensors, so when the dict then evaluated `grad_norms(att)` (reads
   `o.grad`) it hit `None` → the `else 0.0` branch → all zeros (`std` was untouched, so its bars were
   real). `hNorm_attn`/`att` likewise came from the 6-row example forward, not the 144-row training
   batch — undercutting the "even gradient distribution" claim the honest race caption points to.
   **Fix (applied):** capture BOTH arms' training-batch stats *before* any extra forward pass:
   ```python
   def snapshot(step, losses, accs, elapsed):
       hs_std, gs_std = std.last_stats["hNorm"], grad_norms(std)
       ha, aa, ga = att.last_stats["hNorm"], att.last_stats["att"], grad_norms(att)   # capture FIRST
       av = attn_vis(att, ex_rows)                                                     # then clobber freely
       ...
       return { ..., "hNorm_std": hs_std, "gradNorm_std": gs_std, "hNorm_attn": ha,
                "gradNorm_attn": ga, "att": aa, "att_tok": token_alphas(att, ex_rows), ... }  # α LAST
   ```
   **Status:** deployed + server restarted; run 27 now persists real non-zero `gradNorm_attn`
   (L1 0.0028 … L6 4.6e-8) and a crisp look-back head (`attn_vis` L3 h3 `look_w` 0.9993).

2. **Induction only holds on the training distribution (20-distinct, length-21).** Short/random tapes
   degrade the look-back head (~58% at len 8; ~30% on randomised full-length with repeats). Hence the
   21-length fixed tape in the game and the dropped "weekdays" deck (7 symbols can't fill 20 distinct).

3. **"Both arms see the identical stream" is approximate** — same pool/seed/init/schedule, but each arm
   draws its own batch. Fair, just don't over-claim it.

4. **The replay head re-picks per checkpoint** (best look-back head each frame), so layer/head index moves
   early. The card caption now says so; the *behaviour* is the point, not a fixed head.

5. **Owner auto-rerun footgun** — an owner tab with "auto re-run" on will chain runs indefinitely (skips
   cooldown+quota). Disarm it when you don't want to keep burning the GPU.

6. **`/api/gpu/play` uses `run_demo.att or MODEL`** — during a live run `run_demo.att` is the *previous*
   completed model (it's set only when `run_demo` returns). Harmless, but the game isn't reading the
   in-flight model.

---

## 7. Operational commands

```bash
# restart the backend (run the two halves in SEPARATE tool calls; a bg launch hangs a single shell line)
pkill -f "[g]pu_demo_server"                                   # bracket trick: don't match your own arg
GPU_DEMO_OWNER_IPS=67.164.48.33 setsid nohup python3 -u gpu_demo_server.py \
    >/tmp/opencode/gpu_server.log 2>&1 </dev/null &            # run this as its own call

# sanity / verify without a browser
curl -s .../api/gpu/status
curl -s ".../api/gpu/start?dur=30" -H "Origin: https://ai-research-pub-carlos.patriawan-ai.stream"
curl -s ".../api/gpu/play?mode=word&v=64&seed=1"

# JS syntax gate for the §11 script (block index 1 = the §11 IIFE)
python3 -c "import re;print(re.findall(r'<script>(.*?)</script>',open('attnres-deep-dive.html').read(),re.S)[1])" \
    >/tmp/opencode/blk1.js && node --check /tmp/opencode/blk1.js

# publish (byte-identical), then bust the cache link by +1
cp attnres-deep-dive.{html,md} ../ai-research-pub/attention/ && cmp ...
```

---

## 8. Run log

The persisted state (`.gpu_last_model.pt` + `.gpu_last_snap.json`) is overwritten by every
completed run, so a run that finishes *worse* is exactly what the live page then shows. Records
below are the persisted snapshots, newest last.

| run | window | steps | acc std / attn | loss std / attn | ‖h‖ last std / attn | game (6 rows) |
|-----|--------|-------|----------------|-----------------|---------------------|---------------|
| 31  | 180 s  | 2,541 | 1.000 / **0.111** | — / 3.50       | — / —               | weak arm — this is the checkpoint the page showed *before* run 32 |
| 32  | 180 s  | 7,067 | 1.000 / **1.000** | 1.26e-4 / 3.16e-4 | **32.85 / 18.07** | **6/6** correct, p≈0.9999 |

Run 32 internals (for the §11 prose): depth-α (batch/token-mean final aggregation) → **L3 99.72%**,
emb ~0.02%; sharpest look-back head → **L3 head 0**, `look_w` **0.9998** on the answer position
(token-ints sample row 1: position 6 = answer `60`); head-crystallization `look_w` **0.097 @ step 25
→ 1.00** by the end; race to 100% train-acc → std **step 2,575**, attn **step 3,500**.

Note the run-to-run spread in the acc column (run 31's AttnRes arm only reached 11%): the race on
this 6-op toy is high-variance, which is exactly why the page's claim is about *bounded ‖h‖*, not
about the AttnRes arm winning the sprint.
