import json

from cutrain.paths import REPO
from cutrain.serving.serve_adapter import parse_reply, stream_chunks, to_template

DATA = REPO / "data"
READ_FILES = [{"type": "function", "function": {"name": "read_files", "parameters": {
    "type": "object", "properties": {"files": {"type": "array"}}, "required": ["files"]}}}]
# Verbatim from out/app-builds/dnd-music-bot-cline.serve.jsonl: no </parameter> after the array
CLINE_REPLY = ("<think></think>\n<tool_call>\n<function=read_files>\n<parameter=files>\n"
               "[{\"path\": \"/home/pooh/model-apps/dnd-music-bot-cline\", \"start_line\": 1, \"end_line\": 100}]\n"
               "</function>\n</tool_call>\n<|im_end|>")


def test_lenient_recovers_unclosed_parameter_and_strict_does_not():
    _, _, calls, problems = parse_reply(CLINE_REPLY, READ_FILES)
    assert json.loads(calls[0]["function"]["arguments"]) == {}
    assert problems == ["read_files missing ['files']"]
    _, _, calls, problems = parse_reply(CLINE_REPLY, READ_FILES, lenient=True)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["files"][0]["end_line"] == 100
    assert problems == ["read_files unclosed <parameter>"]


def test_lenient_keeps_closed_parameters_before_the_unclosed_one():
    text = ("<tool_call>\n<function=edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=old>\nx = 1\n"
            "</function>\n</tool_call>")
    _, _, calls, _ = parse_reply(text, [], lenient=True)
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py", "old": "x = 1"}


def test_lenient_matches_strict_when_every_parameter_is_closed():
    # a closed value may quote the call syntax itself (writing docs about this format)
    text = ("<tool_call>\n<function=write>\n<parameter=content>\nuse <parameter=x> then <parameter=y>\n</parameter>\n"
            "</function>\n</tool_call>")
    strict, lenient = parse_reply(text, []), parse_reply(text, [], lenient=True)
    assert strict[2][0]["function"]["arguments"] == lenient[2][0]["function"]["arguments"]
    assert strict[3] == lenient[3] == ["unknown tool write"]


def test_reasoning_reaches_the_prompt_within_a_turn_only():
    from transformers import AutoTokenizer
    snapshot = json.loads((DATA.parent / "out/long-grpo-recovery/base_model.json").read_text())["snapshot"]
    tok = AutoTokenizer.from_pretrained(snapshot)
    call = {"id": "call_1", "type": "function", "function": {"name": "read_files", "arguments": "{\"files\": []}"}}
    msgs = [{"role": "user", "content": "build it"},
            {"role": "assistant", "content": None, "reasoning_content": "PLAN-A first", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
    assert to_template(msgs)[1]["reasoning_content"] == "PLAN-A first"
    assert "PLAN-A first" in tok.apply_chat_template(to_template(msgs), tools=READ_FILES, tokenize=False)
    later = msgs + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}]
    assert "PLAN-A first" not in tok.apply_chat_template(to_template(later), tools=READ_FILES, tokenize=False)


def test_stream_sends_reasoning_before_content():
    resp = {"id": "x", "created": 0, "model": "m", "usage": {},
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi", "reasoning_content": "why"}}]}
    deltas = [c["choices"][0]["delta"] for c in stream_chunks(resp)]
    assert deltas[1] == {"reasoning_content": "why"} and deltas[2] == {"content": "hi"}
