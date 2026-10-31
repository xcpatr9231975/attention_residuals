---
name: attention-residuals-gpu-demo
description: Maintains the live GPU demo in §11 of attnres-deep-dive.html (gpu_train.py + gpu_demo_server.py) — the loss race, ‖h‖/gradient bars, α heatmaps, attention matmul replay, the "beat the model" /api/gpu/play game, the copy-head crystalliser, and the honest race scoreboard. Use when editing, restarting, debugging, or re-baking that demo, or when a panel shows wrong/zero numbers.
license: CC-BY-NC-ND-4.0
metadata:
  paper: "2603.15031"
  artifacts: "attnres-deep-dive.html, attnres-deep-dive.md, attnres-deep-dive-implementation.md, gpu_train.py, gpu_demo_server.py"
  companion: "the attention-residuals reference skill holds the paper knowledge; this one holds the demo's operating knowledge"
---

# AttnRes live GPU demo — maintenance skill

The explainer's §11 is a **real** two-model GPU demo, not a sim: `TinyGPT("std")` (PreNorm residual
sum) vs `TinyGPT("attn")` (AttnRes, `DepthAgg` softmax over depth), trained **interleaved** on the same
stream by `gpu_train.run_demo`, served by `gpu_demo_server.py`, drawn by one self-contained HTML page.
**Read `attnres-deep-dive-implementation.md` first** — it is the full architecture + data-flow twin.

## Files

- `~/projects/ai-research-pub/gpu_train.py` — task, both models, `run_demo`, probes (`attn_vis`,
  `token_alphas`, `analyze`, `examples`, `replay_payload`). **Single source of truth** for `K=20,
  VOCAB=64, POOL_N=60000, BATCH=144, SEED=777, INIT_SEED=1234, DURATION=180, CADENCE=25`. Sequence
  length is always `K+1=21`. `run_demo`/`build_model` default to `d=64, dff=128, ×3` (6 depth-ops).
- `~/projects/ai-research-pub/gpu_demo_server.py` — loopback `127.0.0.1:8477`; routes + hardening +
  persistence. Calls `run_demo(duration, emit, stop, ds)` with NO hyper-param override, so the served
  run's size is governed entirely by the `gpu_train` constants above.
- `~/projects/attention_residuals/attnres-deep-dive.{html,md}` — the page + paper twin (published copies
  live in `ai-research-pub/attention/`).

## The task (so panels make sense)

Vectorised induction recall: `k` pairwise-distinct tokens, pick `j∈[0,k-2]`, copy `tok[j]` into the final
position, label = `tok[j+1]` (predict the successor of the repeated token's first occurrence). Fresh
144-batch every step → no memorisation. `tok` = uniform ids `1..64`; `word` = the same rule with the 20
context words drawn **Zipf-weighted without replacement** from the 64-noun `WORDS`. The game's "decks"
(words/int/notes/letters/trap) are just id→label faces — the model only ever sees ints.

## Invariants / gotchas (learned the hard way)

1. **Edit the backend → RESTART the server.** A stale `gpu_demo_server.py` process silently ignores new
   query params (bit me once: a `v` cap was ignored because the process predated the edit).
2. **Restart = two separate tool calls.** `pkill -f "[g]pu_demo_server"` (bracket trick so your own arg
   doesn't match) in one call; the `setsid nohup python3 -u gpu_demo_server.py … &` launch in a *second*
   call — backgrounding inside a single shell line hangs the session.
3. **`start`/`stop` need an Origin/Referer header** (`GPU_DEMO_HOSTS` allow-list, else 403). `curl`-testing
   them requires `-H "Origin: https://ai-research-pub-carlos.patriawan-ai.stream"`.
4. **`/api/gpu/play` answers from `run_demo.att or MODEL`** — during a run that is the *previous*
   completed model; between runs it is the persisted `.gpu_last_model.pt`. `ready:false` → "start a run".
   Boot `load_model()` can OOM under the shared worker (transient); a completed run repopulates it via
   `save_model()`. d64 weights are ~1 MB, so it loads fine once the allocator has any headroom.
5. **Induction is OOD-sensitive:** reliable only on the 20-distinct/length-21 training tape. Short or
   randomly-repeated tapes degrade the look-back head (~58% at len 8). Keep the game tape at 21 tokens;
   do NOT re-add a 7-symbol "weekdays" deck.
6. **The replay head re-picks per checkpoint** (best look-back head each frame); layer/head index moves
   early. The card caption says so — the *behaviour* is the point.
7. **Owner auto-rerun chains forever** (skips cooldown+quota). Disarm the toggle when idle.
8. **Publish = `cp` `.html`+`.md` to `ai-research-pub/attention/`, then `cmp`,** and bump the `?v=N`
   cache-buster on the shared link.
9. **The public page is verified off-box** (Playwright can't reach loopback): use the public URL, and
   verify via `browser_evaluate` + `getImageData` (canvas painted) + `browser_console_messages` — page/element
   screenshots time out (occluded).
10. **Coexists with the resident vLLM worker on the shared GPU.** `BATCH=144` keeps a run's activation
    footprint to a few tens of MB so it fits beside the always-on `VLLM::Worker` (~95.6 GB it holds);
    pushing `BATCH` back toward 384 OOMs under that headroom. `d=64` is the **floor** for a crisp
    look-back head — at `d≤48` the induction head never forms (verified: d48/b192 → look_w 0.22). Both
    arms stay **interleaved, not concurrent**; the only true GPU concurrency is *demo ‖ qwen*.

## ✅ Fixed bug — AttnRes `gradNorm_attn` read as all-zeros

`run_demo.snapshot()` used to call `attn_vis(att, ex_rows)` at its **top**; that `no_grad` forward
reassigns `att.last_outs` to gradless tensors, so the dict's later `grad_norms(att)` read `o.grad is None`
→ **`gradNorm_attn` reported `[0,0,0,0,0,0]`** (fake-zeros AttnRes gradient panel; `gradNorm_std` was
real), and `hNorm_attn`/`att` came from the 6-row example forward, not the 144-batch — undercutting the
"even gradient distribution" claim the honest race caption relies on. Applied fix = capture the
training-batch stats for BOTH arms *before* any extra forward:

```python
def snapshot(step, losses, accs, elapsed):
    hs, gs = std.last_stats["hNorm"], grad_norms(std)                                  # capture FIRST
    ha, aa, ga = att.last_stats["hNorm"], att.last_stats["att"], grad_norms(att)        # (both arms)
    av = attn_vis(att, ex_rows)                                                          # then clobber freely
    ...
    return { ..., "hNorm_std": hs, "gradNorm_std": gs, "hNorm_attn": ha,
             "gradNorm_attn": ga, "att": aa, "att_tok": token_alphas(att, ex_rows), ... }  # α LAST
```

The regression rule: **never read training-batch-derived stats after running an inference forward**
(`analyze`/`attn_vis`/`token_alphas` all overwrite `last_stats`/`last_outs`). **Status:** deployed + server
restarted; run 27 persists real non-zero `gradNorm_attn` (L1 0.0028 … L6 4.6e-8) and a crisp look-back
head (`attn_vis` L3 h3 `look_w` 0.9993).

## Client (§11 script) shape

One IIFE. `el{}` caches ids; `if (!el.gpuLoss) return;`. State: `S{hist,acc,hStd,hAtt,gStd,gAtt,att,attTok,
attnVis,running,done,source,poll,cooldown,runsLeft,owner,auto,histRun,showStd,showAttn,replay}` and
`P{mode,data,guess,you,bot,busy,rpidx,rpTimer}`. `apply(snap)` is the single sink (rebuilds the whole curve
from `snap.hist`, adopts `snap.replay`, resets on new `run_id`, calls `draw()`). SSE `listen()` → poll
fallback (`/api/gpu/snapshot`). Baked fallbacks `DATA_FB/DATA_FB_WORD/ATTN_FB/ATTK_FB/SNAP_FB` render with
the server offline. `playReveal` MUST declare its own `last = d.tokens.length-1` (was `playRender`-scoped)
or the copy-beam + target underline silently vanish.

## Commands

```bash
# restart (two separate calls!)
pkill -f "[g]pu_demo_server"
GPU_DEMO_OWNER_IPS=67.164.48.33 setsid nohup python3 -u gpu_demo_server.py \
    >/tmp/opencode/gpu_server.log 2>&1 </dev/null &
# verify
curl -s .../api/gpu/status
curl -s ".../api/gpu/start?dur=30" -H "Origin: https://ai-research-pub-carlos.patriawan-ai.stream"
curl -s ".../api/gpu/play?mode=word&v=64&seed=1"
# re-bake the 5 fallbacks after a good run: snap->/tmp/opencode/snap_rNN.json, tok+word data->data_rNN.json
python3 /tmp/opencode/bake_r11.py      # rewrites SNAP_FB/ATTN_FB/ATTK_FB/DATA_FB/DATA_FB_WORD in place
# JS gate for the §11 script (regex block index 1)
python3 -c "import re;print(re.findall(r'<script>(.*?)</script>',open('attnres-deep-dive.html').read(),re.S)[1])" \
    >/tmp/opencode/blk1.js && node --check /tmp/opencode/blk1.js
# publish
cp attnres-deep-dive.{html,md} ../ai-research-pub/attention/ && cmp attnres-deep-dive.html ../ai-research-pub/attention/attnres-deep-dive.html
```

## Public URL

`https://ai-research-pub-carlos.patriawan-ai.stream/attention/attnres-deep-dive.html` (bust with `?v=N`).
Tunnel `ai-research-pub-carlos` (`~/.cloudflared/config.research.yml`): `/api/gpu*`→8477, else→8097.
