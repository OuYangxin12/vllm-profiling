#!/usr/bin/env python3
"""
TTFT / TPOT 基准测试脚本
========================
测试条件（对应 task.txt）:
  - qwen3-30b-a3b FP8
  - input:  8192 tokens (8K)
  - output: 1024 tokens (1K)
  - batch size: 4（4 个请求同时并发）

数据获取原理:
  - 使用 OpenAI 兼容 API (stream=True) 流式接收 token
  - TTFT (Time To First Token):  请求发出时刻 -> 收到第一个 token 的延迟
  - TPOT (Time Per Output Token): (收到最后一个 token 时刻 - 收到第一个 token 时刻) / (输出 token 数 - 1)
"""

import asyncio
import json
import random
import time

import aiohttp

API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "qwen3-30b-a3b"
INPUT_LEN = 8192     # 8K 输入
OUTPUT_LEN = 1024    # 1K 输出
BATCH_SIZE = 4       # 并发 batch size


def build_token_ids(target_tokens: int) -> list:
    """直接构造精确 target_tokens 个 token id（避开特殊 token 区间）。

    使用 prompt_token_ids 参数发送，可保证输入 token 数精确，
    避免文本 tokenize 造成的长度偏差。
    """
    random.seed(1234 + target_tokens)
    return [random.randint(1000, 40000) for _ in range(target_tokens)]


async def bench_one(session: aiohttp.ClientSession, req_id: int, token_ids: list, max_tokens: int = OUTPUT_LEN):
    """发送单个流式请求并统计 TTFT / TPOT / token 数。"""
    payload = {
        "model": MODEL_NAME,
        "prompt": token_ids,  # 直接传 token id 列表，保证输入长度精确
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # 强制生成满 output_len，避免 EOS 截断影响 TPOT
    }
    send_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    completion_tokens = 0
    prompt_tokens = 0

    async with session.post(API_URL, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"req {req_id} HTTP {resp.status}: {text[:300]}")
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("text") or ""
            if first_token_time is None and delta:
                first_token_time = time.perf_counter()
            if delta:
                last_token_time = time.perf_counter()

    ttft = first_token_time - send_time
    # TPOT: 首 token 之后每个输出 token 的平均耗时（不含 prefill）
    tpot = (last_token_time - first_token_time) / max(completion_tokens - 1, 1)
    return {
        "req_id": req_id,
        "prompt_tokens": prompt_tokens,
        "ttft_s": ttft,
        "tpot_ms": tpot * 1000,
        "completion_tokens": completion_tokens,
        "total_time_s": last_token_time - send_time,
    }


async def main():
    random.seed(42)
    prompts = [build_token_ids(INPUT_LEN) for _ in range(BATCH_SIZE)]

    print(f"模型: {MODEL_NAME} | 输入 tokens: {INPUT_LEN} | 输出 tokens: {OUTPUT_LEN} | 并发: {BATCH_SIZE}")
    print("-" * 72)

    # 先预热一次，排除 CUDA graph / cache 冷启动影响
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 预热只生成 8 个 token，避免占用 profiler 采集窗口（防 OOM 与窗口浪费）
        warm = await bench_one(session, -1, build_token_ids(64), max_tokens=8)
        print(f"[预热] 完成 (TTFT={warm['ttft_s']*1000:.1f}ms, tokens={warm['completion_tokens']})")
        print("-" * 72)

        batch_start = time.perf_counter()
        results = await asyncio.gather(*(bench_one(session, i, p) for i, p in enumerate(prompts)))
        batch_elapsed = time.perf_counter() - batch_start

    # ---------- 输出结果 ----------
    print(f"{'req':>4} | {'in tokens':>9} | {'TTFT (ms)':>10} | {'TPOT (ms)':>10} | {'out tokens':>10} | {'总耗时 (s)':>10}")
    print("-" * 82)
    for r in results:
        print(f"{r['req_id']:>4} | {r['prompt_tokens']:>9} | {r['ttft_s']*1000:>10.1f} | {r['tpot_ms']:>10.2f} | "
              f"{r['completion_tokens']:>10} | {r['total_time_s']:>10.2f}")
    print("-" * 82)

    ttfts = [r["ttft_s"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    print(f"batch 完成 总耗时: {batch_elapsed:.2f} s")
    print(f"TTFT  平均: {sum(ttfts)/len(ttfts)*1000:.1f} ms | 各请求: {[f'{t*1000:.1f}' for t in ttfts]}")
    print(f"TPOT  平均: {sum(tpots)/len(tpots):.2f} ms | 各请求: {[f'{t:.2f}' for t in tpots]}")
    print(f"每请求吞吐: {OUTPUT_LEN / (sum(t['total_time_s'] for t in results)/len(results)):.1f} tok/s")

    with open("/root/bench_result.json", "w") as f:
        json.dump({"config": {"input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
                              "batch_size": BATCH_SIZE, "model": MODEL_NAME},
                   "results": results,
                   "batch_elapsed_s": batch_elapsed,
                   "mean_ttft_ms": sum(ttfts)/len(ttfts)*1000,
                   "mean_tpot_ms": sum(tpots)/len(tpots)}, f, indent=2)
    print("结果已保存: /root/bench_result.json")


if __name__ == "__main__":
    asyncio.run(main())
