"""
Implementation of DeepMind's RETRO model: https://arxiv.org/pdf/2112.04426.pdf

Contents
--------
1) RETRO overview
2) Notes on this implementation
3) TODOs


1) RETRO overview
-----------------

RETRO is a transformer-based autoregressive language model, which uses a much smaller transformer
(i.e. less parameters) compared to alternatives, while achieving comparable performance.
RETRO does this by augmenting the model with a large database of examples from the training data.
During training and inference, RETRO retrieves items from the database based on their similarity
to its input, and incorporates the retrieved items into the forward pass through a cross-attention
mechanism.

The potentials benefits of RETRO:
 * Same performance as other transformer-based large language models, but with a much smaller
   model. The paper reports that a 7.5B parameter RETRO model is on par, and sometimes
   outperforms, strong models with 175B and 280B parameters (Jurrasic-1 and Gopher).
 * A given transformer model can be "Retrofitted". That is, the database and retrieval mechanisms
   can be added to a pretrained model, making the model into a RETRO model. The model is then
   fine-tuned by keeping the pretrained parameters frozen, and training only the new parameters.
   This scheme increases the model's parameter count by less than 10%, and the training is very
   quick. The fine-tuned RETRO model has performance much better than the pretrained model, and
   performs almost as well as a RETRO model trained from scratch.
 * A trained RETRO model can be easily updated to different contexts or tasks simply by updating
   the retrieval database, without needing to change the model's parameters.
 * Adding items to the retrieval database, without modifying the trained RETRO model, improves
   the performance. This means that a model can be improved simply by collecting more data
   and without needing to retrain.


2) Notes on this implementation
-------------------------------
This implementation is done in Python+PyTroch.
The original RETRO implementation was not publicly released by DeepMind (as of the time of writing
this).
We tried to be very close to the RETRO paper, but some details in the paper are missing, so we
needed to fill-in the blanks. Other minor differences are documented inline.
State tensor dimensions generally follow this order (skip items when not applicable):
("state tensors" - inputs to model or results of intermediate calculations, as opposed
  to parameter tensors;  dimensions order in `attention()` function is different)
 0) sequence dimension (also when the tensor is chunked)
 1) neighbor index
 2) chunk index
 3) batch dimension
 4) vector dimension
in inline comments, we designate tensor sizes using this general rule:
  L = sequence length
  L' = sequence length of keys (in attention)
  C = chunk size
  C' = retrieved chunk size (including continuation)
  N = number of neighbors
  H = number of attention heads
  D = generic dimension size
  D' = generic dimension size
  V = vocabulary size
  elipsis (...) = hides batch dimensions



3) TODO
-------
* training loop, parallelization (maybe convert to Jax)
* retrieval preprocessing and lookup
* sampling word-by-word, with cached activations
* retrofitting
* make transformer model more flexible so that it's easy to retrofit different given models
"""

from math import sqrt
from functools import lru_cache, partial
from itertools import chain

import torch
import torch.nn as nn
from torch.nn.functional import pad, softmax, cross_entropy
import opt_einsum


def view_or_reshape(x, *tensor_or_shape):
    """
    change the view of tensor x to the specified shape or to the specified tensor's shape; if
    impossible, reshape (`reshape` incurs a copy whereas `view` does not)
    """
    if isinstance(tensor_or_shape[0], torch.Tensor):
        shape = tensor_or_shape[0].shape
    else:
        shape = tensor_or_shape
    try:
        x = x.view(shape)
    except RuntimeError:
        x = x.reshape(shape) 
    return x


def chunkenize(x, chunk_size, dim=0):
    """
    given a tensor with a sequence dimension, reshape this dimension to two dimensions:
    one for indexing the chunk and one for indexing position within a chunk.
    `dim` is the sequence dimension.
    For example, if `x` is a tensor with shape`[24,1,2,3,4,5]` representing a sequence of
    length `24` and we use `dim=0` and `chunk_size=4`, we will get a tensor with shape
    `[4,6,1,2,3,4,5]`, where the `6` is in the chunk index dimension, and `4` is in the
    within-chunk dimension.

    TODO: maybe we should change the ordering of dimensions of all objects so that we
     don't need to do transpose here. The transpose operation incurs a copy later.
    """
    dim = dim % x.ndim
    return view_or_reshape(x, *x.shape[:dim], -1, chunk_size,
                           *x.shape[(dim + 1):]).transpose(dim, dim + 1)


def unchunk(x, dim=0):
    """
    the inverse of `chunkenize()`
    """
    return view_or_reshape(x.transpose(dim, dim + 1), *x.shape[:dim], -1, *x.shape[(dim + 2):])


@lru_cache(50)
def cached_einsum_path(subscripts, *shapes):
    """
    torch.einsum does not optimize the contraction path in the case of multiplying more
    than two tensors. Therefore, we use the `opt_einsum` library. But, since we don't
    want to calculate the optimal path everytime, we calculate it once and cache the
    result.
    """
    return opt_einsum.contract_expression(subscripts, *shapes)


def einsum(subscripts, *operands):
    """
    this function is like `torch.einsum`, but uses an optimal contraction path.
    see docstring of function `cached_einsum_path()`.
    """
    path = cached_einsum_path(subscripts, *[tuple(o.shape) for o in operands])
    return path(*operands)


def attention(x_q, x_kv, to_q, to_k, to_v, to_out, mask=None, positional_encoding=None,
              flatten_dims=None, dropout=None):
    """
    attention operation.
    very roughly (without mentioning contraction indexes):
        output = to_out(softmax(to_q(x_q)*[to_k(x_kv)+positional_encoding+mask])*to_v(x_kv))

    this function is implemented to allow a more general attention than required for RETRO.
    In particular, it works also without positional embedding, and the batch dimensions can
    be of any shape.
        
    in einsum strings, these are the meaning of index names:
     d,b,c : vector dimensions
     n,m : sequence dimensions
     h : attention head dimension
     ellipsis (...) hides batch, neighbor, and chunk dimensions.

    Parameters
    ----------
    x_q: input sequence of length N, will be converted to queries
    x_kv: other input sequence, of length M. will be converted to keys and values
    to_q: matrix that will convert `x_q` to N queries
    to_k: matrix that will convert `x_kv` to M keys
    to_v: matrix that will convert `x_kv` to M values
    to_out: matrix that will convert the N combined values (post-attention) to the output
        sequence of length N
    mask: (optional) boolean mask of size N-by-M, `True` indicates that a position is masked.
        used for causal masking.
    positional_encoding: (optional) tensor that will be multiplied by the queries and then
        added to the logits (pre-softmax), used for relative positional encoding.
    flatten_dims:
    dropout:
    """
    if positional_encoding is None:
        logits = einsum('dhb,n...b,dhc,m...c->nm...h', to_q, x_q, to_k, x_kv)
    else:
        # as explained elsewhere in this source code, we use the relative positional
        # encoding of transformerXL (https://arxiv.org/pdf/1901.02860.pdf), but without
        # the bias terms.
        # from the RETRO paper:
        #   > Positional logits are obtained as a linear transform of a cosine vector
        #   > computed from (𝑑(𝑖,𝑖′))𝑖,𝑖′, and are added to content logits, as in a regular
        #   > self-attention block.
        # This sentence makes it seem as if they don't use the bias terms (but it's not
        # 100% clear).
        q = einsum('dhb,n...b->nd...h', to_q, x_q)
        content_logits = einsum('nd...h,dhc,m...c->nm...h', q, to_k, x_kv)
        position_logits = einsum('nd...h,nmdh->nm...h', q, positional_encoding)

        # reshape in-case sizes are different. This happens when the content logits
        # contain an additional neigbor dimension.
        dim_diff = position_logits.ndim - content_logits.ndim
        if dim_diff > 0:
            content_logits = view_or_reshape(
                content_logits,
                *content_logits.shape[:2],
                *(1,) * dim_diff,
                *content_logits.shape[2:]
            )
        elif dim_diff < 0:
            position_logits = view_or_reshape(
                position_logits,
                *position_logits.shape[:2],
                *(1,) * (-dim_diff),
                *position_logits.shape[2:]
            )
        logits = content_logits + position_logits

    rescale_logits = False
    if rescale_logits:
        # scale the logits, as in the original attention paper. we add this, even though
        # it's not clear if it is used in RETRO. In the RETRO paper, equation (4) and
        # listing 1, it does not appear.
        logits /= sqrt(to_q.size(0))

    if mask is not None:
        logits.masked_fill_(mask[(slice(None), slice(None), *(None,) * (logits.ndim - 2))],
                            float('-inf'))

    if flatten_dims is not None:
        # this is required for the case of chunked-cross attention, where logit vecrors
        # from different neighbors are concatenated before applying softmax
        logits = view_or_reshape(logits, logits.size(0), -1, *logits.shape[flatten_dims + 1:])
        x_kv = view_or_reshape(x_kv, -1, *x_kv.shape[flatten_dims:])

    weights = softmax(logits, 1)

    if dropout is not None:
        weights = dropout(weights)

    out = einsum('dch,chb,m...b,nm...h->n...d', to_out, to_v, x_kv, weights)
    return out


class RMSNorm(nn.Module):
    """
    RMSNorm has some configurations not implemented here. It's not clear from the
    RETRO paper which configuration is used, so here we implement the simplest form.
    """
    def __init__(self, d, eps=1e-8, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)

        self.eps = eps
        self.dim_norm = d ** (-1. / 2)

        self.scale = nn.Parameter(torch.ones(d, **factory_kwargs))

    def forward(self, x):
        # x: [..., D]
        rms = x.norm(2, dim=-1, keepdim=True) * self.dim_norm  # [..., 1]
        normed = x / (rms + self.eps)  # [..., D]
        scaled = self.scale * normed  # [..., D]
        return scaled


class Attention(nn.Module):
    """
    attention layer, including optional causal mask and relative positional encoding.
    """

    # caches for positional encoding and causal mask, so that we don't need to recompute
    # them every time anew.
    sincos_cache = dict()
    mask_cache = dict()

    def __init__(self, d, d_x_q=None, d_x_kv=None, d_qk=None, d_v=None, num_heads=1,
                 causal=False, offset=0, flatten_dims=None, dropout=0.0):
        """

        Parameters
        ----------
        d: the dimension of the output sequence.
        d_x_q: the dimension of the input sequence used for queries.
        d_x_kv: the dimension of the input sequence used keys and values.
        d_qk: the dimension of the queries and keys.
        d_v: the dimensions of the values.
        num_heads: number of attention heads
        causal: should disallow attention to future tokens?
        offset: offset between query and key positions, for relative positional encoding
        flatten_dims:
        dropout:
        """
        nn.Module.__init__(self)

        assert d is not None
        if d_x_q is None:
            d_x_q = d
        if d_x_kv is None:
            d_x_kv = d
        if d_qk is None:
            d_qk = d // num_heads
            assert d_qk * num_heads == d
        if d_v is None:
            d_v = d // num_heads
            assert d_v * num_heads == d
        d_out = d

        self.to_q = nn.Parameter(torch.randn(d_qk, num_heads, d_x_q))
        self.to_k = nn.Parameter(torch.randn(d_qk, num_heads, d_x_kv))
        self.to_v = nn.Parameter(torch.randn(d_v, num_heads, d_x_kv))
        self.to_out = nn.Parameter(torch.randn(d_out, d_v, num_heads))
        self.for_pos_enc = nn.Parameter(torch.randn(d_qk, num_heads, d_x_kv))

        self.dropout = nn.Dropout(dropout)

        self.d_x_q = d_x_q
        self.d_x_kv = d_x_kv
        self.d_qk = d_qk
        self.d_v = d_v
        self.num_heads = num_heads
        self.offset = offset or 0
        self.causal = causal
        self.flatten_dims = flatten_dims

    def get_causal_mask(self, sz_q, sz_k, offset, device):

        # first, check if mask is already in cache
        key = (sz_q, sz_k, offset)
        if key in self.mask_cache:
            mask = self.mask_cache[key].to(device)

        # otherwise, create mask
        else:
            mask = torch.ones(sz_q, sz_k, device=device, dtype=torch.bool).triu(1 + offset)

            # put in cache, and evict cache if it's too large
            self.mask_cache[key] = mask
            if len(self.mask_cache) > 50:
                self.mask_cache.pop(next(iter(self.mask_cache)))

        return mask  # [sz_q, sz_k]

    def get_pos_enc(self, sz_q, sz_k, offset, device):
        # TODO: for training, we cache `sincos`, which is the position encoding before
        #  applying the linear transform, because the transform is trainable and changes
        #  between batches.
        #  but for eval, we might as well cache the result of the transform.
        #  (however, we need to be careful about this, since what if we put model on eval mode
        #  temporarily, or we modify the model's weights while in eval?...)
        #
        # TODO: we use relative encoding of TransformerXL, but without the bias terms. This
        #  is what I think is implied  by the retro paper, but it's not defined explicitly.
        #  see comment in function `attention()`

        # first, check if `sincos`` is already in cache
        key = (sz_q, sz_k, offset, self.d_x_kv)
        if key in self.sincos_cache:
            sincos, dists = self.sincos_cache[key]
            sincos = sincos.to(device)  # [sz_q+sz_k-1, d_x_kv]
            dists = dists.to(device)  # [sz_q, sz_k]

        # otherwise, create `sincos`, as in TransformerXL (which is the same as in
        # the paper Attention is All You Need).
        else:

            # signed distance between each query position and each key position
            dists = torch.arange(sz_q, device=device)[:, None] - \
                    torch.arange(sz_k, device=device)[None, :] + offset  # [sz_q, sz_k]

            # all unique distances in a flat tensor
            r = torch.arange(dists.min(), dists.max() + 1.0, 1.0, device=device)  # [sz_q+sz_k-1]

            # inverse of frequencies for sinusoid positional encoding
            inv_freq = 1 / (10000 ** (torch.arange(0.0, self.d_x_kv, 2.0, device=device) /
                                      self.d_x_kv))  # [d_x_kv/2]

            # input to sine and cosine functions
            phases = r[:, None] * (inv_freq[None, :])  # [sz_q+sz_k-1, d_x_kv/2]

            # calculate sines and cosines
            sincos = torch.cat([phases.sin(), phases.cos()], dim=-1)  # [sz_q+sz_k-1, d_x_kv]

            # put in cache, and evict cache if it's too large
            self.sincos_cache[key] = (sincos, dists)
            if len(self.sincos_cache) > 50:
                self.sincos_cache.pop(next(iter(self.sincos_cache)))

        # apply linear transform
        pos_enc = einsum('dhb,nb->ndh', self.for_pos_enc,
                         self.dropout(sincos))  # [sz_q+sz_k-1, d_qk, h]

        # reshape so that `pos_enc[i,j]` will reflect relative position of query `i` and
        # key 'j'.
        pos_enc = pos_enc[dists - dists.min()]  # [sz_q, sz_k, d_qk, h]
        return pos_enc  # [sz_q, sz_k, d_qk, h]

    def forward(self, x_q, x_kv=None):
        # x_q: [L, ..., D]
        # x_kv: [L', ..., D']

        if x_kv is None:
            x_kv = x_q

        n, m = x_q.size(0), x_kv.size(0)
        device = x_q.device

        if self.causal:
            mask = self.get_causal_mask(n, m, self.offset, device)  # [L, L']
        else:
            mask = None

        pos_enc = self.get_pos_enc(n, m, self.offset, device)  # [L, L', D, H]

        return attention(x_q, x_kv, self.to_q, self.to_k, self.to_v, self.to_out,
                         mask, pos_enc, self.flatten_dims, self.dropout)  # [L, ..., D]


class AttentionBlock(nn.Module):
    """
    residual attention with normalization.
    can be used for both self- and cross- attention.
    """
    def __init__(self, d, d_kv=None, num_heads=1, causal=False, dropout=0.0):
        nn.Module.__init__(self)
        if d_kv is None:
            d_kv = d
        self.norm = RMSNorm(d)
        self.attention = Attention(d, d_x_kv=d_kv, num_heads=num_heads, causal=causal,
                                   dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # we need this for weight initialization:
        self.to_out = self.attention.to_out

    def forward(self, x_q, x_kv=None):
        # x_q: [L, ..., D]
        # x_kv: [L', ..., D']

        return x_q + self.dropout(self.attention(self.norm(x_q), x_kv))  # [L, ..., D]


class ChunkedCrossAttentionBlock(nn.Module):
    """
    residual chunked cross attention with normalization.
    """
    def __init__(self, chunk_size, d, d_kv, num_heads=1, dropout=0.0):
        nn.Module.__init__(self)
        if d_kv is None:
            d_kv = d
        self.norm = RMSNorm(d)
        self.attention = Attention(d, d_x_kv=d_kv, num_heads=num_heads, offset=chunk_size,
                                   flatten_dims=2)
        self.dropout = nn.Dropout(dropout)

        self.chunk_size = chunk_size

        # we need this for weight initialization:
        self.to_out = self.attention.to_out

    def forward(self, x_q, x_kv):
        # x_q: [L, ..., D]
        # x_kv: [C', N, L//C, ..., D']
        # `x_kv` is already chunked, but the sequence `x_q` is not.
        # `x_q` and `x_kv` are aligned, meaning that they share a position axis without an
        # offset.
        # we partition the sequence `x_q` into chunks.
        # when chunking, we make sure that no `x_q` element can attend a future chunk,
        # otherwise we would break causality.
        # see explanation in RETRO paper.
        x_q_chunked = chunkenize(
            pad(self.norm(x_q)[self.chunk_size - 1:], (0, 0) * (x_q.ndim - 1) +
                (0, self.chunk_size - 1)),
            self.chunk_size)  # [C, L//C, ..., D]
        z = self.attention(x_q_chunked, x_kv)  # [C, L//C , ..., D]
        z = pad(unchunk(z), (0, 0) * (x_q.ndim - 1) + (z.size(0) - 1, 0))[
            :x_q.size(0)]  # [L, ..., D]
        return x_q + self.dropout(z)  # [L, ..., D]


class FeedForwardBlock(nn.Module):
    """
    residual MLP with normalization.
    """
    def __init__(self, d, d_ff, dropout=0.0):
        nn.Module.__init__(self)
        self.norm = RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
            nn.Dropout(dropout)
        )

        # we need this for weight initialization:
        self.to_out = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1].weight

    def forward(self, x):
        # x: [..., D]
        return x + self.mlp(self.norm(x))  # [..., D]


class Retriever:
    """
    currently, this is a dummy retriever. later we will implement a retriever based
    on BERT and approximate-KNN
    """
    def __init__(
            self,
            vocab_size,
            chunk_size,
            continuation_length,
            num_neighbors
    ):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.continuation_length = continuation_length
        self.num_neighbors = num_neighbors

    def retrieve(self, seq):
        """
        temporary dummy function that maintains causality
        """
        # seq [L, ...]
        seq_chunked = chunkenize(seq, self.chunk_size)  # [C, L//C, ...]

        retrieved = torch.randint(
            self.vocab_size,
            (seq_chunked.shape[1],
             self.chunk_size + self.continuation_length, self.num_neighbors,
             *seq_chunked.shape[2:]),
            device=seq.device,
            generator=torch.Generator(device=seq.device).manual_seed(29847892371)
        )  # [L//C, C', N, ...]
        return retrieved.movedim(0, 2)  # [C', N, L//C, ...]


class Encoder(nn.Module):
    def __init__(
            self,
            num_layers,
            d,
            d_ff,
            d_dec,
            num_heads,
            ca_layers,
            chunk_size,
            dropout=0.0
    ):
        nn.Module.__init__(self)

        self.dropout = nn.Dropout(dropout)

        # self-attention blocks
        self.sa = nn.ModuleList([AttentionBlock(d, num_heads=num_heads, dropout=dropout)
                                 for _ in range(num_layers)])

        # cross-attention blocks
        self.ca = nn.ModuleList(
            [AttentionBlock(d, d_dec, num_heads=num_heads, dropout=dropout) if i in ca_layers
             else None for i in range(num_layers)])

        # feed-forward blocks
        self.ff = nn.ModuleList([FeedForwardBlock(d, d_ff, dropout=dropout)
                                 for _ in range(num_layers)])

        # normalization
        self.norm = RMSNorm(d)

        self.chunk_size = chunk_size

        # we will use this for weight initialization of the final projection of each
        # attention/ff layer
        self.for_init = [m for m in chain(self.sa, self.ca, self.ff) if m is not None]

    def forward(self, y, x):
        """
        y: [C', N, L//C, ..., D']
        x: [L, ..., D]
        """
        x = chunkenize(x, self.chunk_size)  # [C, L//C, ..., D]
        for sa, ca, ff in zip(self.sa, self.ca, self.ff):
            y = sa(y)  # [C', N, L//C, ..., D']
            if ca is not None:
                y = ca(y, x)  # [C', N, L//C, ..., D']
            y = ff(y)  # [C', N, L//C, ..., D']
        return self.norm(y)  # [C', N, L//C, ..., D']


class Decoder(nn.Module):
    def __init__(
            self,
            num_layers,
            d,
            d_ff,
            d_enc,
            num_heads,
            ca_start,
            ca_freq,
            chunk_size,
            encoder,
            dropout=0.0):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

        def use_cross(i):
            return (i >= ca_start) and ((i - ca_start) % ca_freq == 0)

        # self-attention blocks
        self.sa = nn.ModuleList(
            [AttentionBlock(d, num_heads=num_heads, causal=True, dropout=dropout)
             for _ in range(num_layers)])

        # cross-attention blocks
        self.ca = nn.ModuleList(
            [ChunkedCrossAttentionBlock(chunk_size, d, d_enc, num_heads=num_heads, dropout=dropout)
             if use_cross(i) else None for i in range(num_layers)])

        # feed-forward blocks
        self.ff = nn.ModuleList([FeedForwardBlock(d, d_ff, dropout=dropout)
                                 for _ in range(num_layers)])
        self.norm = RMSNorm(d)

        self.chunk_size = chunk_size

        # we will use this for weight initialization of the final projection of each
        # attention/ff layer
        self.for_init = [m for m in chain(self.sa, self.ca, self.ff) if m is not None]

    def forward(self, x, y):
        # x: [L, ..., D]
        # y: [C', N, L//C, ..., D']

        z = None

        for sa, ca, ff in zip(self.sa, self.ca, self.ff):
            x = sa(x)  # [L, ..., D]
            if ca is not None:
                if z is None:
                    z = self.encoder(y, x)  # [C', N, L//C, B, D']
                x = ca(x, z)  # [L, ..., D]
            x = ff(x)  # [L, ..., D]
        return self.norm(x)  # [L, ..., D]


class RetroLanguageModel(nn.Module):
    """
    combines RETRO decoder, encoder, embeddings, and output projection to vocab.
    """
    def __init__(
            self,
            vocab_size,
            d_dec,
            d_ff_dec,
            num_heads_dec,
            num_layers_dec,
            ca_start_dec,
            ca_freq_dec,
            d_enc,
            d_ff_enc,
            num_heads_enc,
            num_layers_enc,
            ca_layers,
            chunk_size,
            continuation_length,
            num_neighbors,
            dropout,
            share_embeddings=False
    ):
        nn.Module.__init__(self)

        assert (not share_embeddings) or (d_enc == d_dec)

        self.retriever = Retriever(vocab_size, chunk_size, continuation_length, num_neighbors)
        self.embedding_dec = nn.Embedding(vocab_size, d_dec)
        self.embedding_enc = nn.Embedding(vocab_size, d_enc) if not share_embeddings \
            else partial(self.embedding_dec)
        self.encoder = Encoder(num_layers_enc, d_enc, d_ff_enc, d_dec, num_heads_enc,
                               ca_layers, chunk_size, dropout)
        encoder = partial(self.encoder)
        self.decoder = Decoder(num_layers_dec, d_dec, d_ff_dec, d_enc, num_heads_dec,
                               ca_start_dec, ca_freq_dec, chunk_size, encoder, dropout)
        self.to_vocab = nn.Linear(d_dec, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """
        RETRO paper claims to follow GPT2. In GPT2, weights are all initialized
        using N(0, 0.02^2).
        (the GPT2 paper references the GPT paper, which claims to use N(0, 0.02),
        but looking at the official GPT2 code we see that they actually 0.02 to
        be the std, not variance).

        """
        def xavier_(p):
            nn.init.xavier_normal_(p.view(-1, p.size(-1)), 1.0)

        def normal_(p):
            nn.init.normal_(p, std=0.02)

        init_func = normal_

        def init(module):
            if isinstance(module, RMSNorm):
                assert len(list(module.parameters())) == 1  # scale
                nn.init.constant_(module.scale, 1.0)
            elif isinstance(module, Attention):
                assert len(list(module.parameters())) == 5  # to_q, to_k, to_v, to_out, for_pos_enc
                for p in module.parameters():
                    init_func(p)
            elif isinstance(module, nn.Linear):
                assert len(list(module.parameters())) == 2  # weight, bias
                init_func(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                assert len(list(module.parameters())) == 1  # weight
                init_func(module.weight)

        self.apply(init)

        # from GPT2 paper::
        #   > A modified initialization which accounts for the accumulation on the residual
        #   > path with model depth. We scale the weights of residual layers at initialization
        #   > by a factor of 1/√N where N is the # of residual layers.
        for subnet in [self.decoder, self.encoder]:
            for module in subnet.for_init:
                module.to_out.data /= sqrt(len(subnet.for_init))

    def forward(self, seq, labels=None):

        # retrieve neighbors from database, for each chunk in `seq`
        retrieved = self.retriever.retrieve(seq)  # [C', N, L//C, ...]

        # convert tokens to embeddings
        x = self.dropout(self.embedding_dec(seq))  # [L, ..., D]
        y = self.dropout(self.embedding_enc(retrieved))  # [C', N, L//C, ..., D']

        # apply decoder and language model head
        logits = self.to_vocab(self.decoder(x, y))  # [L, ..., V]

        # calculate and return loss if labels were given
        if labels is not None:
            loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss  # [,]

        else:
            return logits  # [L, ..., V]
