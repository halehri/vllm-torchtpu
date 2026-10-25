# Step-level queue simulator of the vLLM chunked-prefill scheduler for AgentX replays.
# Requests come from an aiperf profile_export.jsonl (profiling phase). Each conversation's turns stay
# sequential: turn k+1 arrives at sim_end(k) + measured client gap. Step time is priced by the path
# the step takes: decode-only = AD + CD*n_dec; prefill-carrying = AP + BP*prefill_tokens + CD*n_dec.
# Policies: paw (warmest-first admission, running in admission order), srpt (all prefill candidates
# shortest-remaining first, longest held back), cap=C (prefill tokens <= C while decodes run),
# srpt_min=N (srpt hold-back skipped while >= N decodes run).
import json, sys, argparse, collections, heapq, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('profile'); ap.add_argument('--policy', default='paw')
ap.add_argument('--cap', type=int, default=0); ap.add_argument('--srpt-min', type=int, default=0)
ap.add_argument('--srpt-wait', type=float, default=0.0)  # SRPT only while the oldest waiter has waited > X s
ap.add_argument('--srpt-qlen', type=int, default=0)  # SRPT only while >= Q requests wait
ap.add_argument('--mbt', type=int, default=8192); ap.add_argument('--seats', type=int, default=32)
ap.add_argument('--al', type=float, default=3.32); ap.add_argument('--k', type=int, default=3)
ap.add_argument('--ad', type=float, default=13.0); ap.add_argument('--cd', type=float, default=0.9)
ap.add_argument('--ap', type=float, default=31.0); ap.add_argument('--bp', type=float, default=0.047)
a = ap.parse_args()

# --- workload
conv = collections.defaultdict(list)
for line in open(a.profile):
    r = json.loads(line); m = r['metadata']; x = r['metrics']
    if m.get('benchmark_phase') != 'profiling': continue
    g = lambda k: (x.get(k) or {}).get('value') if isinstance(x.get(k), dict) else x.get(k)
    isl = g('usage_prompt_tokens') or 0; hit = g('usage_prompt_cache_read_tokens') or 0
    conv[m['conversation_id']].append(dict(start=m['request_start_ns'] / 1e6, end=m['request_end_ns'] / 1e6,
                                           isl=isl, comp=max(isl - hit, 1), osl=max(g('output_token_count') or 1, 1)))
t0 = min(t['start'] for L in conv.values() for t in L)
chains = []
for L in conv.values():
    L.sort(key=lambda t: t['start'])
    for i, t in enumerate(L):
        t['gap'] = max(0.0, L[i + 1]['start'] - t['end']) if i + 1 < len(L) else None
    chains.append(L)

# --- simulation
arrivals = []  # (time, chain idx, turn idx)
for ci, L in enumerate(chains):
    heapq.heappush(arrivals, (L[0]['start'] - t0, ci, 0))
waiting = []; running = []  # running: dicts with rem_prefill, rem_out, admit_order
now = 0.0; order = 0; done = []
def step_time(T, nd):
    return (a.ap + a.bp * T + a.cd * nd) if T > 0 else (a.ad + a.cd * nd)
while arrivals or waiting or running:
    while arrivals and arrivals[0][0] <= now:
        t, ci, ti = heapq.heappop(arrivals)
        req = dict(chains[ci][ti]); req.update(ci=ci, ti=ti, arr=t, rem=req['comp'], out=0.0, first=None)
        waiting.append(req)
    if not waiting and not running:
        now = arrivals[0][0]; continue
    decodes = [r for r in running if r['rem'] == 0]
    nd = len(decodes)
    budget = a.mbt - nd * (1 + a.k)
    capped = a.cap > 0 and nd > 0
    if capped: budget = min(budget, a.cap)
    # prefill candidates
    run_pf = [r for r in running if r['rem'] > 0]
    seats = a.seats - len(running)
    if a.policy == 'paw':
        cand = sorted(run_pf, key=lambda r: r['order'])
        wq = sorted(waiting, key=lambda r: r['rem'])[:64] if waiting else []
        wq = sorted(wq, key=lambda r: r['rem']) + [r for r in sorted(waiting, key=lambda r: r['arr']) if r not in wq]
    else:  # srpt
        cand = list(run_pf); wq = sorted(waiting, key=lambda r: r['rem'])
    sched = []  # (req, tokens)
    if a.policy == 'paw':
        for r in cand:
            if budget <= 0: break
            c = min(r['rem'], budget, a.cap if capped else 1 << 30); sched.append((r, c)); budget -= c
        for r in wq:
            if budget <= 0 or seats <= 0: break
            c = min(r['rem'], budget, a.cap if capped else 1 << 30); sched.append((r, c)); budget -= c; seats -= 1
    else:
        hold = not (a.srpt_min and nd >= a.srpt_min)
        if a.srpt_wait and waiting:
            hold = hold and (now - min(r['arr'] for r in waiting)) > a.srpt_wait * 1e3
        elif a.srpt_wait:
            hold = False
        if a.srpt_qlen:
            hold = hold and len(waiting) >= a.srpt_qlen
        pool = cand + (wq[:max(seats, 0)] if hold else [])
        pool.sort(key=lambda r: r['rem'])
        if not hold:  # plain admission-order for running, then waiting
            pool = sorted(cand, key=lambda r: r['order']) + wq[:max(seats, 0)]
        for r in pool:
            if budget <= 0: break
            if r in waiting and seats <= 0: continue
            c = min(r['rem'], budget, a.cap if capped else 1 << 30); sched.append((r, c)); budget -= c
            if r in waiting: seats -= 1
    T = sum(c for _, c in sched)
    dt = step_time(T, nd); now += dt
    for r, c in sched:
        if r in waiting:
            waiting.remove(r); r['order'] = order; order += 1; running.append(r)
        r['rem'] -= c
        if r['rem'] == 0: r['first'] = now; r['out'] = 1.0
    for r in decodes:
        r['out'] += a.al
    fin = [r for r in running if r['rem'] == 0 and r['out'] >= r['osl']]
    for r in fin:
        running.remove(r); r['fin'] = now; done.append(r)
        L = chains[r['ci']]
        if r['ti'] + 1 < len(L): heapq.heappush(arrivals, (now + L[r['ti']]['gap'], r['ci'], r['ti'] + 1))
ttft = np.array([r['first'] - r['arr'] for r in done])
inter = np.array([(r['osl'] - 1) / max(r['fin'] - r['first'], 1e-3) * 1e3 for r in done if r['osl'] > 1])
toks = sum(r['isl'] + r['osl'] for r in done)
print(f"policy={a.policy} cap={a.cap} srpt_min={a.srpt_min}: n={len(done)} makespan={now/1e3:.0f}s "
      f"tok/s={toks/now*1e3:.0f} p90_ttft={np.percentile(ttft,90):.0f}ms p50_ttft={np.median(ttft):.0f}ms "
      f"p90_inter={np.percentile(inter,10):.1f} median_inter={np.median(inter):.1f}")
