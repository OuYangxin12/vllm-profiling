#!/usr/bin/env python3
"""v3 nsys 采集方案验证客户端：warmup + batch4(8K入/128出)，触发 prefill 与 decode 两阶段。"""
import asyncio
import json
import os
import random
import time

import aiohttp

API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "qwen3-30b-a3b"
INPUT_LEN = 8192
OUTPUT_LEN = int(os.environ.get("BENCH_OUTPUT_LEN", 128))   # nc 基准参数为 1024：BENCH_OUTPUT_LEN=1024
BATCH_SIZE = 4


def build_token_ids(target: int) -> list:
    random.seed(1234 + target)
    return [random.randint(1000, 40000) for _ in range(target)]


async def bench_one(session, req_id, token_ids, max_tokens=OUTPUT_LEN):
    payload = {"model": MODEL_NAME, "prompt": token_ids, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}, "ignore_eos": True}
    send = time.perf_counter()
    first = last = None
    n = 0
    async with session.post(API_URL, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(await resp.text())
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = (json.loads(line[len("data:"):])["choices"] or [{}])[0].get("text")
            if delta:
                t = time.perf_counter()
                first = first or t
                last = t
                n += 1
    return {"req_id": req_id, "ttft_s": first - send, "tpot_ms": (last - first) / max(n - 1, 1) * 1000,
            "completion_tokens": n, "total_time_s": last - send}


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        warm = await bench_one(s, -1, build_token_ids(64), max_tokens=8)
        print(f"[warmup] TTFT={warm['ttft_s']*1000:.1f}ms tokens={warm['completion_tokens']}", flush=True)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*(bench_one(s, i, build_token_ids(INPUT_LEN)) for i in range(BATCH_SIZE)))
        print(f"[bench] elapsed={time.perf_counter()-t0:.2f}s mean_ttft={sum(r['ttft_s'] for r in rs)/4*1000:.1f}ms "
              f"mean_tpot={sum(r['tpot_ms'] for r in rs)/4:.2f}ms", flush=True)
    out_path = os.environ.get("BENCH_OUT", "/work/verify_bench_result.json")
    with open(out_path, "w") as f:
        json.dump(rs, f, indent=2)
    print(f"saved {out_path}", flush=True)


asyncio.run(main())
