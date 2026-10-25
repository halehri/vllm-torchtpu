# AgentX scheduler queue simulator

`agentx_queue_sim.py` replays an aiperf `profile_export.jsonl` (profiling phase) through a step-level
model of vLLM's chunked-prefill scheduler. The model keeps each conversation's turns sequential, with
the measured client gaps between them, and prices every engine step by the path it takes:

- decode-only steps cost `AD + CD * n_decodes`;
- prefill-carrying steps cost `AP + BP * prefill_tokens + CD * n_decodes`.

It reports tok/s, p90 TTFT and p90 interactivity (p10 per-user tok/s) per policy:

- `--policy paw`: warmest-first admission, with running requests in admission order;
- `--cap C`: per-step prefill cap while decodes run;
- `--policy srpt`: shortest-remaining-first with hold-back, gated by `--srpt-min N`, `--srpt-wait X`
  or `--srpt-qlen Q`.

Calibration on Qwen3.8-Flash c32 (2x2x1, MTP K=3, AL 3.32): `--ap 50 --bp 0.042`. This reproduces the
measured baseline tok/s and p90 interactivity, with p90 TTFT 24% low. It gets the sign of the
measured SRPT and step-cap arms right, with magnitudes off by up to ~40%. Use it to rank policies
before queuing arms, not as an exact predictor.

    python3 tools/sched_sim/agentx_queue_sim.py <profile_export.jsonl> --policy srpt --srpt-min 16 --ap 50 --bp 0.042
