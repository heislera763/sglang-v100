import argparse
import os
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def redact(value):
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if key in ("api_key", "admin_api_key") else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--lengths", type=int, nargs="+", default=[1000, 8192, 25000])
p.add_argument("--output-len", type=int, default=1024)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--output-dir", default=".")
a = p.parse_args()
root = Path(a.output_dir)
root.mkdir(parents=True, exist_ok=True)
out = root / f"{a.tag}_measurements.jsonl"
assert not out.exists()
url = "http://127.0.0.1:9001"
session = requests.Session()
for line in Path("/home/alexander/.llama-server/.env").read_text().splitlines():
    if line.startswith("LLAMA_API_KEY="):
        session.headers["Authorization"] = "Bearer " + line.split("=", 1)[
            1
        ].strip().strip("\"'")

for _ in range(900):
    try:
        if session.get(url + "/health", timeout=5).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError("server not ready")
tokenizer = AutoTokenizer.from_pretrained(
    os.environ.get(
        "MODEL_PATH",
        "/home/alexander/.llama-server/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4",
    )
)
for length in a.lengths:
    for repeat in range(a.repeats + 1):
        seed = 20260830 + max(0, repeat - 1)
        n = 256 if repeat == 0 else a.output_len
        task = [
            "Write a detailed practical guide of at least 2000 words explaining how to investigate a slow web service. Cover reproducible measurements, CPU and memory costs, databases, concurrency, and ways to verify improvements. Use concrete examples and explain the reasoning behind each step.",
            "Write a detailed Python tutorial of at least 2000 words on building a bounded in-memory cache. Include implementation examples, eviction policies, concurrency, expiration, testing, and operational tradeoffs. Explain each design decision and walk through realistic examples.",
        ][seed % 2]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Task: " + task}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        note = tokenizer.encode(
            "Reference note: Software services process requests using finite CPU time, memory, storage, and network capacity. Useful measurements separate queueing time from work time and record both typical and slow requests. A repeatable workload and a clear comparison help identify whether a change actually improves service.\n",
            add_special_tokens=False,
        )
        separator = tokenizer.encode("\n\n", add_special_tokens=False)
        padding = length - len(ids) - len(separator)
        assert padding >= 0
        # Qwen's chat template begins with start-of-message, user role, newline.
        assert tokenizer.decode(ids[:3]).endswith("user\n")
        ids = (
            ids[:3]
            + (note * ((padding + len(note) - 1) // len(note)))[:padding]
            + separator
            + ids[3:]
        )
        assert len(ids) == length
        session.post(url + "/flush_cache", timeout=60).raise_for_status()
        session.post(
            url + "/set_internal_state", json={"server_args": {}}, timeout=60
        ).raise_for_status()
        start = time.perf_counter()
        times = []
        counts = []
        metas = []
        with session.post(
            url + "/generate",
            json=dict(
                input_ids=ids,
                sampling_params=dict(temperature=0, max_new_tokens=n, ignore_eos=True),
                stream=True,
            ),
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(chunk_size=None):
                if not line.startswith(b"data: "):
                    continue
                if line[6:] == b"[DONE]":
                    break
                item = json.loads(line[6:])
                meta = item["meta_info"]
                count = meta["completion_tokens"]
                if not counts or count > counts[-1]:
                    times.append(time.perf_counter() - start)
                    counts.append(count)
                    metas.append(
                        {k: v for k, v in meta.items() if k.startswith("spec")}
                    )
        assert (
            counts[-1] == n
            and meta["prompt_tokens"] == length
            and meta.get("cached_tokens") == 0
        ), (counts[-1], meta)
        windows = []
        first = 0
        for j in range(1, len(counts)):
            if counts[j] - counts[first] >= 256 or j == len(counts) - 1:
                windows.append(
                    dict(
                        tokens=counts[j] - counts[first],
                        tok_s=(counts[j] - counts[first]) / (times[j] - times[first]),
                        start_count=counts[first],
                        end_count=counts[j],
                    )
                )
                first = j
        info = session.get(url + "/get_server_info", timeout=60).json()
        (root / f"{a.tag}_server_info.json").write_text(
            json.dumps(redact(info), indent=2) + "\n"
        )
        row = dict(
            scenario="natural_tutorial",
            task=task,
            tag=a.tag,
            input_len=length,
            output_len=n,
            seed=seed,
            warmup=repeat == 0,
            ttft_s=times[0],
            prefill_tok_s=length / times[0],
            server_prefill_tok_s=(
                length / meta["server_ttft"] if meta.get("server_ttft", 0) > 0 else None
            ),
            server_decode_tok_s=meta.get("stream_decode_throughput"),
            decode_tok_s=(counts[-1] - counts[0]) / (times[-1] - times[0]),
            times_s=times,
            completion_tokens=counts,
            windows=windows,
            meta_info=meta,
            spec_timeline=metas,
            output_text=item.get("text", ""),
        )
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            a.tag,
            length,
            n,
            seed,
            "prefill",
            round(row["prefill_tok_s"], 2),
            "decode",
            round(row["decode_tok_s"], 3),
            "slowest",
            round(min(w["tok_s"] for w in windows), 3),
            "spec",
            metas[-1],
            flush=True,
        )
(root / f"{a.tag}_bench.done").write_text("done")
