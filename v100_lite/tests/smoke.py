"""Exercise the 9001 API contract; never access production port 9000."""

import argparse, base64, io, json, os
from pathlib import Path
import requests
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--tag")
p.add_argument("--disabled-plugin", action="store_true")
args = p.parse_args()
if args.disabled_plugin:
    import sys
    os.environ.pop("SGLANG_V100_LITE", None)
    import sglang_v100_lite
    sglang_v100_lite.register()
    assert "sglang_v100_lite.runtime" not in sys.modules
    print("Plugin disabled: no runtime or kernel imports")
    raise SystemExit(0)
root = Path(__file__).resolve().parents[2]
s = requests.Session()
key = os.environ.get("API_KEY") or os.environ.get("LLAMA_API_KEY")
if key:
    s.headers["Authorization"] = "Bearer " + key
url = "http://127.0.0.1:9001"
results = {}


def post(path, payload):
    r = s.post(url + path, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


model = s.get(url + "/v1/models", timeout=15).json()["data"][0]["id"]


def chat(messages, **kw):
    payload = dict(
        model=model,
        messages=messages,
        max_completion_tokens=512,
        temperature=0,
        chat_template_kwargs={"enable_thinking": False},
    )
    payload.update(kw)
    return post("/v1/chat/completions", payload)


results["models"] = model
results["tokenize"] = post(
    "/tokenize", {"prompt": "What is six times seven?", "reasoning_effort": "xhigh"}
)
results["sampling"] = chat(
    [{"role": "user", "content": "Answer with only the number: six times seven."}],
    temperature=0.6,
    top_p=0.95,
)
assert "42" in (results["sampling"]["choices"][0]["message"].get("content") or ""), (
    results["sampling"]
)
results["xhigh"] = chat(
    [{"role": "user", "content": "What is six times seven?"}],
    reasoning_effort="xhigh",
    chat_template_kwargs={"enable_thinking": True},
)
reasoning_message = results["xhigh"]["choices"][0]["message"]
assert "42" in (reasoning_message.get("content") or ""), results["xhigh"]
assert reasoning_message.get("reasoning_content"), results["xhigh"]
img = Image.new("RGB", (112, 112), (0, 255, 0))
buf = io.BytesIO()
img.save(buf, format="PNG")
content = [
    {"type": "text", "text": "Name the dominant color in this image in one word."},
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        },
    },
]
results["image"] = chat([{"role": "user", "content": content}])
assert (
    "green" in (results["image"]["choices"][0]["message"].get("content") or "").lower()
), results["image"]
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]
results["tool"] = chat(
    [{"role": "user", "content": "Use multiply to multiply 6 by 7."}],
    tools=tools,
    tool_choice="required",
)
m = results["tool"]["choices"][0]["message"]
assert m.get("tool_calls"), m
call = m["tool_calls"][0]
assert call["function"]["name"] == "multiply", call
assert json.loads(call["function"]["arguments"]) == {"a": 6, "b": 7}, call
results["tool_roundtrip"] = chat(
    [
        {"role": "user", "content": "Use multiply to multiply 6 by 7."},
        m,
        {"role": "tool", "tool_call_id": call["id"], "content": "42"},
    ],
    tools=tools,
)
assert "42" in (
    results["tool_roundtrip"]["choices"][0]["message"].get("content") or ""
), results["tool_roundtrip"]
chunks = []
payload = dict(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Explain how to measure web server latency in a short paragraph.",
        }
    ],
    max_completion_tokens=128,
    stream=True,
    return_timing_metrics=True,
    temperature=0,
    chat_template_kwargs={"enable_thinking": False},
)
with s.post(
    url + "/v1/chat/completions", json=payload, stream=True, timeout=180
) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
            obj = json.loads(line[6:])
            metrics = obj.get("sglext", {}).get("timing_metrics")
            if metrics:
                chunks.append(metrics)
assert any(x.get("server_ttft", 0) > 0 for x in chunks), chunks
assert any(x.get("stream_decode_throughput", 0) > 0 for x in chunks), chunks
assert all("prompt_tokens" in x and "completion_tokens" in x for x in chunks), chunks
results["stream_timings"] = chunks
if args.tag:
    output = root / "artifacts" / f"{args.tag}-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
print(
    "PASS",
    args.tag,
    model,
    "sampling, xhigh, tokenize, image, tool roundtrip, streaming server timing fields",
)
