from types import SimpleNamespace
import json
import pytest
from cutrain.dataprep.prepare_full_run import converted
from cutrain.training.train_long_grpo import sft_examples


class Tokenizer:
    eos_token = '<eos>'

    def apply_chat_template(self, messages, **kwargs):
        # Exposes loss of tool definitions or reasoning during conversion.
        prefix = json.dumps(kwargs.get('tools')) + '\n'
        return prefix + ''.join('<|im_start|>' + m['role'] + '\n' +
                              m.get('reasoning_content', '') + m['content'] + '<eos>\n'
                              for m in messages)


def test_conversion_keeps_reasoning_tools_and_single_eos():
    r = dict(prompt=[dict(role='user', content='question')],
             chosen=[dict(role='assistant', content='answer', reasoning_content='reasoning')],
             tools=[{'name': 'example'}])
    prompt, completion = list(converted(r, Tokenizer()))[0]
    assert 'example' in prompt and 'question' in prompt
    assert completion == 'reasoninganswer'
    row = dict(format='rendered_sft_v1', rendered_prompt=prompt, completion=completion)
    assert list(sft_examples(None, SimpleNamespace(), 0, [row])) == [(prompt, completion, 1)]


def test_episode_only_labels_requested_assistant():
    r = dict(messages=[dict(role='user', content='q'), dict(role='assistant', content='prefix'),
                       dict(role='tool', content='evidence'), dict(role='assistant', content='finish')],
             assistant_loss_indices=[3])
    pairs = list(converted(r, Tokenizer()))
    assert len(pairs) == 1
    assert 'evidence' in pairs[0][0] and pairs[0][1] == 'finish'


def test_multiturn_chosen_is_rejected_instead_of_mislabelled():
    with pytest.raises(ValueError):
        list(converted(dict(prompt=[], chosen=[{'role': 'assistant'}, {'role': 'tool'}]), Tokenizer()))
