"""Render a bounded mixed SFT corpus without trimming history or reasoning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random


def converted(row, tok):
    if 'messages' in row:
        messages = row['messages']
        indices = row.get('assistant_loss_indices', [])
        if not indices:
            raise ValueError('episode has no labelled actions')
        from cutrain.serving.serve_adapter import render_turn
        for i in indices:
            if i < 1 or messages[i]['role'] != 'assistant':
                raise ValueError('bad assistant label')
            # Native template rendering of the turn, as for chosen completions below.
            yield render_turn(tok, messages, i, row.get('tools'))
    else:
        chosen = row['chosen']
        if len(chosen) != 1 or chosen[0]['role'] != 'assistant':
            raise ValueError('expected a single assistant completion')
        kw = dict(tools=row.get('tools') or None, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(row['prompt'], **kw) + '<|im_start|>assistant\n'
        full = tok.apply_chat_template(row['prompt'] + chosen, **kw).rstrip('\n')
        if not full.startswith(prompt) or not full.endswith(tok.eos_token):
            raise ValueError('template prefix/EOS mismatch')
        # Preserve the native template's reasoning and tool-call representation.
        yield prompt, full[len(prompt):-len(tok.eos_token)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', required=True, help='Unified mixture selected by build_dataset.py')
    ap.add_argument('--max-context', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.bos_token = None
    dest = Path(args.out)
    if dest.exists():
        raise SystemExit('output already exists')
    inputs = [args.data, 'data/long_agent_sft_memory_v2.jsonl',
              'data/long_agent_stopping_sft.jsonl']
    rows = []
    for path in inputs:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rows.append(r)
    random.Random(args.seed).shuffle(rows)
    seen = set()
    kept = Counter()
    dropped = Counter()
    prepared = []
    for r in rows:
        source = r.get('source', 'unknown')
        ident = (source, str(r.get('id', r.get('episode_id'))))
        if ident in seen:
            dropped[source + ': duplicate'] += 1
            continue
        seen.add(ident)
        try:
            pairs = list(converted(r, tok))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            dropped[source + ': schema/template ' + type(exc).__name__] += 1
            continue
        for prompt, completion in pairs:
            n = len(tok.encode(prompt, add_special_tokens=False)) + len(tok.encode(completion + tok.eos_token, add_special_tokens=False))
            if not completion.strip() or n > args.max_context:
                dropped[source + ': empty/over_context'] += 1
                continue
            prepared.append(dict(format='rendered_sft_v1', source=source, source_id=ident[1],
                                 rendered_prompt=prompt, completion=completion, tokens=n))
            kept[source] += 1
    required = ['hf:v4-distill', 'hf:glm-reasoning', 'hf:openthoughts-agent',
                'verified_local_oracle', 'verified_local_stopping_oracle']
    if any(not kept[s] for s in required):
        raise SystemExit('missing required source after filtering: ' + str(kept))
    random.Random(args.seed).shuffle(prepared)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f:
        for r in prepared:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report = dict(examples=len(prepared), kept_by_source=dict(kept), dropped=dict(dropped),
                  max_context=args.max_context, seed=args.seed, inputs=inputs,
                  max_tokens=max(r['tokens'] for r in prepared))
    dest.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
