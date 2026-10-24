"""Conservative training-time repetition scoring and verified single-turn preferences.

No logits processors or stopping criteria: GRPO samples and scores the same policy.
JSON string values, native tool-call parameter values, fenced code, and
punctuation-only spans are exempt. This intentionally misses some loops rather
than punishing legitimate code/literals.
"""
import math
import re
import json


def protected_ranges(text):
    ranges = [(m.start(), m.end()) for m in re.finditer(r"```[\s\S]*?(?:```|$)", text)]
    # Native calls carry code and file contents as raw parameter text, the counterpart of JSON strings.
    ranges += [(m.start(), m.end()) for m in re.finditer(r"<parameter=[^>]*>[\s\S]*?(?:</parameter>|$)", text)]
    try:
        if isinstance(json.loads(text), (dict, list)):
            ranges.append((0, len(text)))
    except (ValueError, TypeError):
        pass
    # Protect quoted literals only after a JSON-looking object/array opener.
    in_string = escaped = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                ranges.append((start, i + 1))
                in_string = False
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
        elif ch == '"' and depth:
            in_string, start = True, i
    if in_string:
        ranges.append((start, len(text)))
    return ranges


def token_loops(ids, decode, min_tokens=64, max_period=32, min_repeats=4):
    """Find nonoverlapping exact token cycles anywhere in one generated response."""
    ids = list(ids)
    text = decode(ids)
    protected = protected_ranges(text)
    spans = []
    i = 0
    while i < len(ids):
        best = None
        for width in range(1, min(max_period, (len(ids) - i) // min_repeats) + 1):
            count = max(min_repeats, math.ceil(min_tokens / width))
            block = ids[i:i + width]
            end = i + width * count
            if end > len(ids) or ids[i:end] != block * count:
                continue
            while end + width <= len(ids) and ids[end:end + width] == block:
                end += width
            left, right = len(decode(ids[:i])), len(decode(ids[:end]))
            excerpt = text[left:right]
            if not any(c.isalnum() for c in excerpt) or "\ufffd" in excerpt:
                continue
            if any(a <= left and right <= b for a, b in protected):
                continue
            if best is None or end > best["end"]:
                best = {"start": i, "end": end, "period": width,
                        "repeats": (end - i) // width, "excerpt": excerpt[:160]}
        if best:
            spans.append(best)
            i = best["end"]
        else:
            i += 1
    return {"loop_tokens": sum(s["end"] - s["start"] for s in spans), "spans": spans}


def apply_loop_reward(result, records, weight):
    """Cap the episode-level penalty; no reward for merely being short."""
    out = dict(result)
    count = sum(bool(r["loop_tokens"]) for r in records)
    penalty = weight if count else 0.0
    out.update(task_reward=result["reward"], token_loop_turns=count,
               token_loop_tokens=sum(r["loop_tokens"] for r in records),
               token_loop_penalty=penalty, reward=result["reward"] - penalty,
               token_loop_message_indices=[r["message_index"] for r in records if r["loop_tokens"]],
               loop_completions=[{"message_index": r["message_index"], "content": r["completion"]}
                                 for r in records if r["loop_tokens"]],
               truncated_message_indices=[r["message_index"] for r in records if r["truncated"]])
    return out


def loop_preferences(results, transcripts):
    """Observed loop negatives only; verified alternatives must share the full prefix."""
    good = [i for i, r in enumerate(results) if r["success"] and r["verified"] and r["intact"]
            and r["tests"] == 1 and not r.get("token_loop_turns", 0)]
    pairs = []
    for b, bad in enumerate(results):
        for j in bad.get("token_loop_message_indices", []):
            for g in good:
                if j >= len(transcripts[g]) or transcripts[g][:j] != transcripts[b][:j]:
                    continue
                chosen = transcripts[g][j]
                raw = next((x["content"] for x in bad.get("loop_completions", []) if x["message_index"] == j), None)
                if raw is None:
                    continue
                rejected = {"role": "assistant", "content": raw}
                if chosen["role"] != "assistant" or rejected["role"] != "assistant" or chosen == rejected:
                    continue
                pairs.append({"format": "verified_loop_preference_v1", "tools": [],
                              "prompt": transcripts[b][:j], "chosen": [chosen], "rejected": [rejected],
                              "chosen_verification": results[g], "rejected_verification": bad,
                              "rejected_message_index": j,
                              "rejected_eos": j not in bad.get("truncated_message_indices", []),
                              "source": "observed_policy_loop_vs_verified_policy_success"})
                break
    return pairs


def evaluation_metrics(results):
    if not results:
        return {}
    n = len(results)
    return {"episodes": n, "success_rate": sum(r["success"] for r in results) / n,
            "token_loop_episode_rate": sum(r.get("token_loop_turns", 0) > 0 for r in results) / n,
            "action_loop_episode_rate": sum(r["repeats"] > 0 for r in results) / n,
            "abstention_rate": sum(r.get("abstained", False) for r in results) / n,
            "premature_stop_rate": sum(r.get("premature_stop", False) for r in results) / n,
            "truncation_episode_rate": sum(r["truncated_turns"] > 0 for r in results) / n}
