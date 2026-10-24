"""Self-check for the chunked SFT loss: python "computer use training/test_sft_think.py"

The chunked path slices hidden states by one position and drops the prompt before computing
cross-entropy. Get either wrong and the model is trained to predict the wrong token with no error
anywhere, so compare it against the obvious full-logits implementation.
"""
import torch
import torch.nn.functional as F
from torch import nn

from cutrain.training.train_orpo import _chunk_nll

torch.manual_seed(0)
B, T, H, V, CHUNK = 2, 40, 16, 50, 7
lm_head = nn.Linear(H, V, bias=False)
hidden_all = torch.randn(B, T, H)
labels = torch.randint(0, V, (B, T))
labels[:, :25] = -100          # the prompt: masked out, exactly as the collator does
labels[1, -4:] = -100          # padding at the end of the shorter row


def chunked(hidden, labels, chunk):
    """What ChunkedSFTTrainer.compute_loss does."""
    first = int((labels != -100).int().argmax(-1).min())
    hidden, labels = hidden[:, first - 1:-1], labels[:, first:]
    nll = torch.cat([_chunk_nll(lm_head, hidden[:, i:i + chunk], labels[:, i:i + chunk])
                     for i in range(0, labels.shape[1], chunk)], dim=1)
    mask = labels != -100
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def reference(hidden, labels):
    """Full logits, standard causal shift: position t predicts token t+1."""
    logits = lm_head(hidden).float()
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)


got, want = chunked(hidden_all, labels, CHUNK), reference(hidden_all, labels)
assert torch.allclose(got, want, atol=1e-5), f"chunked {got.item()} vs reference {want.item()}"

# chunk size must not change the answer, and a chunk boundary must not drop a token
for chunk in (1, 3, 8, 15, 1000):
    assert torch.allclose(chunked(hidden_all, labels, chunk), want, atol=1e-5), chunk

# the prompt really is excluded: hidden states before the hand-off position cannot move the loss
shuffled = hidden_all.clone()
shuffled[:, :23] = torch.randn(B, 23, H)        # position 24 feeds label 25, so 0..23 are free
assert torch.allclose(chunked(shuffled, labels, CHUNK), want, atol=1e-5)
# but the hand-off position itself does matter: it predicts the first completion token
handoff = hidden_all.clone()
handoff[:, 24] += 1.0
assert not torch.allclose(chunked(handoff, labels, CHUNK), want, atol=1e-4)

# a completion label does move it
moved = labels.clone()
moved[0, 30] = (int(labels[0, 30]) + 1) % V
assert not torch.allclose(chunked(hidden_all, moved, CHUNK), want, atol=1e-4)
print("ok")
