import torch
from model import RetroLanguageModel

vocab_size = 1000
seq_length = 26
batch_size = 3

seq = torch.randint(vocab_size, (seq_length, batch_size))

model = RetroLanguageModel(
    vocab_size,
    d_dec=16,
    d_ff_dec=32,
    num_heads_dec=2,
    num_layers_dec=10,  # in paper: =L
    ca_start_dec=5,  # in paper: =P[0]=6  (paper is 1-based, not zero-based)
    ca_freq_dec=3,  # in paper: implied by P: "The retrieval models contain one Retro-block every 3 blocks,
    d_enc=8,  # in paper: =d'
    d_ff_enc=16,
    num_heads_enc=2,
    num_layers_enc=2,  # in paper: =Lenc=2
    ca_layers_enc={1},  # in paper: =P_enc={1}
    chunk_size=4,  # in paper: =m=64  (this also equals the neighbor chunk size)
    continuation_length=4,  # in paper: =64
    num_neighbors=2,  # in paper: =key
    dropout=0.1,
).eval()

out, _ = model.forward(seq)

## test shape
assert out.shape == (len(out), batch_size, vocab_size)

## test autoregressivity
for r in [1,2,3,4,5,6]:

    out1, _ = model.forward(seq[:-r])
    out2 = model.forward(seq)[0][:-r]

    assert torch.allclose(out1, out2, atol=1e-6, rtol=5e-3)

## test cache
out1, _ = model.forward(seq)
model.reset_cache().using_cache().updating_cache()
for i in range(len(seq)):
    out2_, _ = model.forward(seq[[i]])
    out1_ = out1[[i]]

    assert torch.allclose(out1_, out2_, atol=1e-6, rtol=5e-3)
