"""OpenAI-compatible chat server for a trained LoRA adapter, so real agent harnesses (opencode) can drive it.

Tools are rendered by the model's own chat template (the native <tool_call><function=...> format the
SFT data uses) and parsed back into OpenAI tool_calls. One request generates at a time. Every turn is
appended to --log so a test can score tool-call validity per turn, not only per task.

  python -m cutrain.serving.serve_adapter --adapter out/<run>/grpo-long/adapter --port 8011 --log out/serve.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from cutrain.evaluation.probe_agent import load
from cutrain.training.train_orpo import ASSISTANT

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>[ \t]*\n?(.*?)\n?[ \t]*</parameter>", re.S)
# Serving only: a parameter also ends at a next <parameter= line or at </function>, as in vLLM's qwen3_coder
# parser. The model drops </parameter> after JSON-array values (Cline's read_files); training stays strict.
LENIENT_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>[ \t]*\n?(.*?)(?:\n?[ \t]*</parameter>|(?=\n<parameter=)|\n?[ \t]*$)", re.S)


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def to_template(messages: list[dict]) -> list[dict]:
    """OpenAI request messages -> the chat template's schema (arguments as dicts, plain-text content).

    reasoning_content passes through: the template keeps it after the last user message (this turn's tool
    steps) and empties it before, so the model sees its own earlier reasoning within a turn.
    """
    out = []
    for m in messages:
        msg = {"role": m["role"], "content": text_of(m.get("content"))}
        if m["role"] == "assistant" and isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip():
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"].get("arguments") or "{}"
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args = {"raw": args}
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": c["function"]["name"], "arguments": args}})
            msg["tool_calls"] = calls
        if m["role"] == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def render_turn(tok, messages: list[dict], i: int, tools: list[dict] | None) -> tuple[str, str]:
    """(prompt, completion) for assistant message i, both cut from one chat-template rendering.

    The completion is the turn exactly as the template writes it (reasoning, content, native calls),
    without the closing EOS line; callers append EOS.
    """
    prompt = tok.apply_chat_template(to_template(messages[:i]), tools=tools or None, tokenize=False) + ASSISTANT
    full = tok.apply_chat_template(to_template(messages[:i + 1]), tools=tools or None, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("rendered turn is not a continuation of the prompt")
    if not full.endswith(tok.eos_token + "\n"):
        raise ValueError("unexpected end of rendered turn")
    return prompt, full[len(prompt):-len(tok.eos_token + "\n")]


# The template prints scalar arguments with Python's str(), so booleans and null come back in this spelling.
PY_LITERALS = {"True": True, "False": False, "None": None}


def convert(raw: str, schema: dict | None):
    """Parameter text -> value. Strings stay strings; other schema types are JSON-decoded when possible."""
    kind = (schema or {}).get("type")
    if kind == "string":
        return raw
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return PY_LITERALS.get(raw.strip(), raw)


def parse_reply(text: str, tools: list[dict], lenient: bool = False) -> tuple[str, str, list[dict], list[str]]:
    """Completion -> (content, reasoning, tool_calls, problems with the calls).

    lenient also accepts parameters missing </parameter> (still listed in problems). Only when the strict
    parse leaves a <parameter= unmatched, so a closed value quoting that text parses as in strict mode.
    """
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    think = THINK_RE.search(text)
    body = THINK_RE.sub("", text)
    calls, problems = [], []
    for name, inner in FUNC_RE.findall(body):
        props = schemas.get(name, {}).get("properties", {})
        params = PARAM_RE.findall(inner)
        if lenient and "<parameter=" in PARAM_RE.sub("", inner):
            problems.append(f"{name} unclosed <parameter>")
            params = LENIENT_PARAM_RE.findall(inner)
        args = {k: convert(v, props.get(k)) for k, v in params}
        if name not in schemas:
            problems.append(f"unknown tool {name}")
        missing = [k for k in schemas.get(name, {}).get("required", []) if k not in args]
        if missing:
            problems.append(f"{name} missing {missing}")
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if "<tool_call>" in body and not calls:
        problems.append("malformed tool_call block")
    content = CALL_RE.sub("", body).replace("<|im_end|>", "").strip()
    return content, think.group(1).strip() if think else "", calls, problems


class Model:
    def __init__(self, args):
        self.args = args
        self.model, self.tok = load(args)
        self.lock = threading.Lock()

    @torch.no_grad()
    def chat(self, body: dict) -> dict:
        tools = body.get("tools") or []
        msgs = to_template(body["messages"])
        prompt = self.tok.apply_chat_template(msgs, tools=tools or None, tokenize=False) + ASSISTANT
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)
        budget = min(int(body.get("max_tokens") or body.get("max_completion_tokens") or self.args.max_new_tokens),
                     self.args.max_new_tokens)
        if ids.shape[1] + budget > self.args.max_context:
            raise ValueError(f"prompt {ids.shape[1]} tokens + {budget} exceeds max context {self.args.max_context}")
        temperature = float(body.get("temperature", self.args.temperature) or 0)
        started = time.monotonic()
        with self.lock:
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=budget,
                                      pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
                                      **({"do_sample": False} if temperature <= 0 else
                                         {"do_sample": True, "temperature": temperature, "top_p": 0.95}))
        completion = out[0, ids.shape[1]:]
        text = self.tok.decode(completion, skip_special_tokens=False)
        truncated = int(completion[-1]) != self.tok.eos_token_id if completion.numel() else True
        content, reasoning, calls, problems = parse_reply(text, tools, lenient=True)
        if truncated:
            problems.append("hit max_tokens before ending the turn")
        record = {"time": time.time(), "prompt_tokens": ids.shape[1], "completion_tokens": int(completion.numel()),
                  "seconds": round(time.monotonic() - started, 2), "tools_offered": len(tools),
                  "tool_calls": [c["function"]["name"] for c in calls], "problems": problems, "raw": text}
        with open(self.args.log, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        message = {"role": "assistant", "content": content or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model", "local"),
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if calls else ("length" if truncated else "stop")}],
                "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": int(completion.numel()),
                          "total_tokens": ids.shape[1] + int(completion.numel())}}


def stream_chunks(resp: dict):
    """A finished completion as OpenAI SSE chunks (reasoning, content, then each tool call, then the finish).

    reasoning_content goes out as its own delta (DeepSeek's field) so clients can send it back next request.
    """
    base = {k: resp[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    msg, choice = resp["choices"][0]["message"], resp["choices"][0]
    yield base | {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    if msg.get("reasoning_content"):
        yield base | {"choices": [{"index": 0, "delta": {"reasoning_content": msg["reasoning_content"]},
                                   "finish_reason": None}]}
    if msg.get("content"):
        yield base | {"choices": [{"index": 0, "delta": {"content": msg["content"]}, "finish_reason": None}]}
    for i, call in enumerate(msg.get("tool_calls", [])):
        yield base | {"choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": i, "id": call["id"], "type": "function", "function": call["function"]}]}}]}
    yield base | {"choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
                  "usage": resp["usage"]}


def handler(model: Model, name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            print(f"{self.address_string()} {fmt % a}", flush=True)

        def send_json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self.send_json(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"}]})
            else:
                self.send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self.send_json(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                resp = model.chat(body)
            except Exception as exc:  # noqa: BLE001 - report to the client instead of dropping the socket
                print(f"request failed: {type(exc).__name__}: {exc}", flush=True)
                return self.send_json(400, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            if not body.get("stream"):
                return self.send_json(200, resp)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in stream_chunks(resp):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="LoRA adapter; omit for the base model")
    ap.add_argument("--name", default="cu", help="model id reported to clients")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.0, help="used when the client sends none")
    ap.add_argument("--log", required=True, help="JSONL of every generated turn")
    ap.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kernels", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    model = Model(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(model, args.name))
    print(f"serving {args.adapter or 'base model'} as '{args.name}' on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
