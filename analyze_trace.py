#!/usr/bin/env python3
"""
torch profiler chrome trace 分析脚本：统计 prefill / decode 两阶段 kernel 构成
用法: python3 analyze_trace.py <trace.json.gz 或 .json>
prefill/decode 判据: user_annotation 步骤时长 > 200ms 视为 prefill chunk
"""
import gzip, json, sys, glob
from collections import defaultdict

fn = sys.argv[1] if len(sys.argv) > 1 else sorted(
    [f for f in glob.glob('prof_traces/*.pt.trace.json.gz') if 'warmup' not in f])[-1]
opener = gzip.open if fn.endswith('.gz') else open
with opener(fn) as f:
    evs = json.load(f)['traceEvents']

ann = sorted([e for e in evs if e.get('cat') == 'user_annotation'], key=lambda e: e['ts'])
kerns = sorted([e for e in evs if e.get('cat') == 'kernel'], key=lambda e: e['ts'])

prefill_steps = [a for a in ann if a['dur'] > 200_000]
decode_steps = [a for a in ann if a['dur'] <= 200_000 and a['name'].startswith('execute_context_0')]
print(f"trace: {fn}")
print(f"prefill chunks: {len(prefill_steps)}, decode steps: {len(decode_steps)}")
pf_s, pf_e = prefill_steps[0]['ts'], prefill_steps[-1]['ts'] + prefill_steps[-1]['dur']
dc_s, dc_e = decode_steps[0]['ts'], decode_steps[-1]['ts'] + decode_steps[-1]['dur']


def summarize(sel, label, wall):
    agg = defaultdict(lambda: [0, 0.0])
    for e in sel:
        agg[e['name']][0] += 1
        agg[e['name']][1] += e['dur']
    total_us = sum(v[1] for v in agg.values())
    print(f"\n=== {label}: {len(sel)} kernels | GPU busy {total_us/1e6:.2f}s / wall {wall:.2f}s "
          f"({total_us/1e6/wall*100:.0f}%) ===")
    for name, (cnt, dur) in sorted(agg.items(), key=lambda x: -x[1][1])[:12]:
        print(f"  {dur/1e3:9.1f}ms {cnt:7d}x  {name[:88]}")


summarize([e for e in kerns if pf_s <= e['ts'] <= pf_e], "PREFILL", (pf_e - pf_s) / 1e6)
summarize([e for e in kerns if dc_s <= e['ts'] <= dc_e], "DECODE", (dc_e - dc_s) / 1e6)
durs = [a['dur'] / 1000 for a in decode_steps]
print(f"\ndecode step: 平均 {sum(durs)/len(durs):.1f}ms  min {min(durs):.1f}  max {max(durs):.1f}")
