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

The potential benefits of RETRO:
 * Same performance as other transformer-based large language models, but with a much smaller
   model. The paper reports that a 7.5B parameter RETRO model is on par, and sometimes
   outperforms, strong models with 175B and 280B parameters (Jurrasic-1 and Gopher).
   Smaller model means simpler and faster training.
   The retrieval database requires large memory (RAM / SSD), but this is cheaper than GPU.
 * A given transformer model can be "Retrofitted". That is, the database and retrieval mechanisms
   can be added to a pretrained model, making the model into a RETRO model. The model is then
   fine-tuned by keeping the pretrained parameters frozen, and training only the new parameters.
   This scheme increases the model's parameter count by less than 10%, and the training is very
   quick. The fine-tuned RETRO model has performance much better than the pretrained model, and
   performs almost as well as a RETRO model trained from scratch.
 * A trained RETRO model can be easily updated to different contexts or tasks simply by updating
   the retrieval database, without needing to change the model's parameters.
 * Adding items to the retrieval database, without modifying the trained RETRO model, improves
   the performance. This means that a model can be improved ("learn new knowledge") simply by
   collecting more data and without needing to retrain.


2) Notes on this implementation
-------------------------------
This implementation is done in Python+PyTorch.

The original RETRO implementation was not publicly released by DeepMind (as of the time of writing
this).
We tried to be very close to the RETRO paper, but some details in the paper are missing, so we
needed to fill-in the blanks. See inline documentation.

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
  batch dimension is hidden behind ellipsis (...)
  (different occurrences of D might designate different numbers. the idea is mostly to track
   different types of dimensions...)

3) TODO
-------
* training loop, parallelization (maybe convert to Jax)
* retrieval preprocessing and lookup
* retrofitting
* make transformer model more flexible so that it's easy to retrofit different given models
"""

from math import sqrt
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
from torch.nn.functional import softmax, cross_entropy

from utils import *


def chunkenize(x, chunk_size):
    """
    given a tensor with a sequence dimension, reshape this dimension to two dimensions:
    one for indexing the chunk and one for indexing position within a chunk.
    For example, if `x` is a tensor with shape`[24,5]` representing a sequence of
    length `24` and we use `chunk_size=4`, we will get a tensor with shape
    `[4,6,5]`, where the `6` is in the chunk index dimension, and `4` is in the
    within-chunk dimension.
    return also the remainder of `x` that didn't fit into a full chunk.

    """
    t = len(x) - (len(x) % chunk_size)
    residue = x[t:]
    x = x[:t]
    if len(x) == 0:
        chunked = None
    else:
        chunked = torch.reshape(x, (-1, chunk_size, *x.shape[1:])).transpose(0, 1)
    if len(residue) == 0:
        residue = None
    return chunked, residue


def unchunk(x):
    """
    the inverse of `chunkenize()`
    """
    return torch.reshape(x.transpose(0, 1), (-1, *x.shape[2:]))


def attention(q, k, v, mask=None, positional_encoding=None, additional_dim=None, dropout=None,
              rescale_logits=False):
    """
    attention operation.

    this function is implemented to allow a more general attention than required for RETRO:
    it works also without positional embedding, and the batch dimensions can be of any shape.

    in einsum strings, these are the meaning of index names:
     d : vector dimensions
     n,m : sequence dimensions
     h : attention head dimension
     ellipsis (...) hides batch, neighbor, and chunk dimensions.

    Parameters
    ----------
    q: query sequence of length L, for each head
    k: key sequence of length L', for each head
    v: value sequence of length L', for each head
    mask: (optional) boolean mask of size L-by-L', `True` indicates that a position is masked.
        used for causal masking.
    positional_encoding: (optional) tensor that will be multiplied by the queries and then
        added to the logits (pre-softmax), used for relative positional encoding.
    additional_dim:
    dropout:
    rescale_logits:
    """

    content_logits = torch.einsum('n...hd,m...hd->nm...h', q, k)

    if dont_have(positional_encoding):
        position_logits = 0.0
    else:
        # we use the relative positional encoding of transformerXL
        # (https://arxiv.org/pdf/1901.02860.pdf), but without the bias terms.
        # from the RETRO paper:
        #   > Positional logits are obtained as a linear transform of a cosine vector
        #   > computed from (𝑑(𝑖,𝑖′))𝑖,𝑖′, and are added to content logits, as in a regular
        #   > self-attention block.
        # This sentence makes it seem as if they don't use the bias terms (but it's not
        # 100% clear).
        position_logits = torch.einsum('n...hd,nmhd->nm...h', q, positional_encoding)

        # this is required for the case of chunked-cross attention, where content logit contain
        # an additional neighbor dimension
        if have(additional_dim):
            position_logits.unsqueeze_(additional_dim)

    logits = content_logits + position_logits

    if rescale_logits:
        # scale the logits, as in the original attention paper. we add this, even though
        # it's not clear if it is used in RETRO. In the RETRO paper, equation (4) and
        # listing 1, it does not appear.
        logits /= sqrt(q.size(-1))

    if have(mask):
        logits.masked_fill_(mask[(slice(None), slice(None), *(None,) * (logits.ndim - 2))],
                            float('-inf'))

    if have(additional_dim):
        # this is required for the case of chunked-cross attention, where logit vectors
        # from different neighbors are concatenated before applying softmax
        logits = logits.flatten(additional_dim - 1, additional_dim)
        v = v.flatten(end_dim=additional_dim - 1)

    weights = softmax(logits, 1)

    if have(dropout):
        weights = dropout(weights)

    return torch.einsum('nm...h,m...hd->n...hd', weights, v)


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


class Attention(nn.Module, ModuleWithCache):
    """
    attention layer, including optional causal mask and relative positional encoding.
    """

    # caches for positional encoding and causal mask, so that we don't need to recompute
    # them every time anew.
    sincos_cache = dict()
    mask_cache = dict()

    def __init__(self, d, d_x_q=None, d_x_kv=None, d_qk=None, d_v=None, num_heads=1,
                 causal=False, offset=0, additional_dim=None, dropout=0.0, use_kv_cache=False, use_offset_cache=False):
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
        additional_dim:
        dropout:
        """
        nn.Module.__init__(self)
        ModuleWithCache.__init__(self)

        assert have(d)
        if dont_have(d_x_q):
            d_x_q = d
        if dont_have(d_x_kv):
            d_x_kv = d
        if dont_have(d_qk):
            d_qk = d // num_heads
            assert d_qk * num_heads == d
        if dont_have(d_v):
            d_v = d // num_heads
            assert d_v * num_heads == d
        d_out = d

        def parameter(*sz):
            return nn.Parameter(torch.randn(*sz))

        self.to_q = parameter(num_heads, d_qk, d_x_q)
        self.to_kv = parameter(num_heads, d_qk + d_v, d_x_kv)
        self.for_pos_enc = parameter(num_heads, d_qk, d_x_kv)
        self.to_o = parameter(d_out, num_heads, d_v)

        self.dropout = nn.Dropout(dropout)

        self.d_x_q = d_x_q
        self.d_x_kv = d_x_kv
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_out = d_out
        self.num_heads = num_heads
        self.offset = offset or 0
        self.causal = causal
        self.additional_dim = additional_dim
        self.use_kv_cache = use_kv_cache
        self.use_offset_cache = use_offset_cache

    def get_causal_mask(self, sz_q, sz_k, offset, device):

        # first, check if mask is already in cache
        key = (sz_q, sz_k, offset)
        if key in self.mask_cache:
            mask = self.mask_cache[key].to(device)  # [sz_q, sz_k]

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
        pos_enc = torch.einsum('hdb,nb->nhd', self.for_pos_enc,
                               self.dropout(sincos))  # [sz_q+sz_k-1, h, d_qk]

        # reshape so that `pos_enc[i,j]` will reflect relative position of query `i` and
        # key 'j'.
        pos_enc = pos_enc[dists - dists.min()]  # [sz_q, sz_k, h, d_qk]
        return pos_enc  # [sz_q, sz_k, h, d_qk]

    def forward(self, x_q, x_kv):
        # x_q: [L, ..., D]
        # x_kv: [L', ..., D']

        q = torch.einsum('hdb,n...b->n...hd', self.to_q, x_q)  # [L, ..., H, D]

        if have(x_kv):
            kv = torch.einsum('hdb,m...b->m...hd', self.to_kv, x_kv)  # [L', ..., H, D+D']
            k = kv[..., :self.d_qk]  # [L', ..., H, D]
            v = kv[..., self.d_qk:]  # [L', ..., H, D']
        else:
            k = None
            v = None

        offset_fix = 0
        if self.use_kv_cache:
            kv_cache = self.get_from_cache('kv')
            if have(kv_cache):
                k_cache, v_cache = kv_cache

                if dont_have(k):
                    k = k_cache.to(q.device)
                    v = v_cache.to(q.device)
                else:
                    k = torch.cat((k_cache.to(k.device), k))
                    v = torch.cat((v_cache.to(v.device), v))

            if have(k):
                self.update_cache((k, v), 'kv')
            else:
                self.update_cache(None, 'kv')

            if self.use_offset_cache:
                offset_fix_cache = self.get_from_cache('offset_fix')
                if dont_have(offset_fix_cache):
                    offset_fix_cache = 0
                offset_fix = offset_fix_cache
                self.update_cache(offset_fix + len(q), 'offset_fix')

        if dont_have(k):
            return torch.zeros(*x_q.shape[:-1], self.d_out, device=self.to_o.device, dtype=self.to_o.dtype)

        n, m = len(q), len(k)

        if self.causal:
            mask = self.get_causal_mask(n, m, self.offset + offset_fix, q.device)  # [L, L']
        else:
            mask = None

        pos_enc = self.get_pos_enc(n, m, self.offset + offset_fix, q.device)  # [L, L', H, D]

        attn = attention(q, k, v, mask, pos_enc, self.additional_dim, self.dropout)  # [L, ..., H, D]

        out = torch.einsum('dhb,n...hb->n...d', self.to_o, attn)  # [L, ..., D]

        return out


class AttentionBlock(nn.Module, ModuleWithCache):
    """
    residual attention with normalization.
    can be used for both self- and cross- attention.
    """

    def __init__(self, d, d_kv=None, num_heads=1, causal=False, dropout=0.0, use_kv_cache=False):
        nn.Module.__init__(self)
        ModuleWithCache.__init__(self)

        if dont_have(d_kv):
            d_kv = d
        self.norm = RMSNorm(d)
        self.attention = Attention(d, d_x_kv=d_kv, num_heads=num_heads, causal=causal,
                                   dropout=dropout, use_kv_cache=use_kv_cache, use_offset_cache=use_kv_cache)
        self.dropout = nn.Dropout(dropout)

        # we need this for weight initialization:
        self.to_out = [self.attention.to_o, ]

        self.register_modules_with_cache(self.attention)

    def forward(self, x_q, x_kv=None):
        # x_q: [L, ..., D]
        # x_kv: [L', ..., D']

        x_q_normed = self.norm(x_q)

        if dont_have(x_kv):
            x_kv = x_q_normed

        return x_q + self.dropout(self.attention(x_q_normed, x_kv))  # [L, ..., D]


class ChunkedCrossAttentionBlock(nn.Module, ModuleWithCache):
    """
    residual chunked cross attention with normalization.
    """

    def __init__(self, chunk_size, d, d_kv, num_heads=1, dropout=0.0):
        nn.Module.__init__(self)
        ModuleWithCache.__init__(self)

        if dont_have(d_kv):
            d_kv = d
        self.norm = RMSNorm(d)
        self.attention = Attention(d, d_x_kv=d_kv, num_heads=num_heads, offset=chunk_size - 1,
                                   additional_dim=2, use_kv_cache=True, use_offset_cache=False)
        self.dropout = nn.Dropout(dropout)

        self.chunk_size = chunk_size

        # we need this for weight initialization:
        self.to_out = [self.attention.to_o, ]

        self.register_modules_with_cache(self.attention)

    @staticmethod
    def get_stuff(pos, length, chunk_size):
        num_elements_to_omit = min(max(0, chunk_size - 1 - pos), length)
        num_elements_to_keep = length - num_elements_to_omit
        len_x_q_padded = chunk_size * (((pos + length) // chunk_size) - (max(0, pos - (chunk_size - 1)) // chunk_size))
        # len_x_q_padded = ((num_elements_to_keep + chunk_size - 1) // chunk_size) * chunk_size
        ind1_x_q = num_elements_to_omit
        ind1_x_q_padded = (num_elements_to_omit + pos + 1) % chunk_size
        ind2_x_q_padded = ind1_x_q_padded + num_elements_to_keep
        ind1_z_padded = ind1_x_q
        ind2_z_padded = ind1_z_padded + num_elements_to_keep

        return \
            len_x_q_padded, \
            ind1_x_q, \
            ind1_x_q_padded, \
            ind2_x_q_padded, \
            ind1_z_padded, \
            ind2_z_padded

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

        pos = 0
        pos_cached = self.get_from_cache('pos')
        if have(pos_cached):
            pos = pos_cached
        self.update_cache((pos + len(x_q)), 'pos')

        # if dont_have(x_kv):
        #     return x_q

        chunk_size = self.chunk_size

        len_x_q_padded, ind1_x_q, ind1_x_q_padded, \
        ind2_x_q_padded, ind1_z_padded, ind2_z_padded = self.get_stuff(pos, len(x_q), chunk_size)

        x_q_padded = torch.zeros((len_x_q_padded, *x_q.shape[1:]), device=x_q.device, dtype=x_q.dtype)
        x_q_padded[ind1_x_q_padded:ind2_x_q_padded] = x_q[ind1_x_q:]
        x_q_chunked, _ = chunkenize(x_q_padded, chunk_size)  # [C, L//C, ..., D]
        if dont_have(x_q_chunked):
            return x_q
        z_chunked = self.attention(x_q_chunked, x_kv)  # [C, L//C , ..., D]
        if (pos + len(x_q) + 1) % chunk_size == 0:
            self.attention.update_cache(None, 'kv')
        else:
            kv_cache = self.attention.get_from_cache('kv')
            if have(kv_cache):
                slc = (slice(None), slice(None), slice(-1, None), ...)
                self.attention.update_cache((kv_cache[0][slc], kv_cache[1][slc]), 'kv')
        z = unchunk(z_chunked)
        z_padded = torch.zeros_like(x_q)
        z_padded[ind1_z_padded:ind2_z_padded] = z[ind1_x_q_padded:ind2_x_q_padded]
        return x_q + self.dropout(z_padded)  # [L, ..., D]


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
        self.to_out = [[m for m in self.mlp.modules() if isinstance(m, nn.Linear)][-1].weight, ]

    def forward(self, x):
        # x: [..., D]
        return x + self.mlp(self.norm(x))  # [..., D]


class Retriever(ModuleWithCache):
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
        ModuleWithCache.__init__(self)

        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.continuation_length = continuation_length
        self.num_neighbors = num_neighbors

    def retrieve(self, seq):
        """
        temporary dummy function that maintains causality
        """
        # seq [L, ...]
        last_partial_chunk = self.get_from_cache('last_partial_chunk')
        if have(last_partial_chunk):
            seq = torch.cat((last_partial_chunk.to(seq.device), seq))

        seq_chunked, last_partial_chunk = chunkenize(seq, self.chunk_size)  # [C, L//C, ...]

        rng = torch.Generator(device=seq.device).manual_seed(29847892371)
        rng_state = self.get_from_cache('rng_state')
        if have(rng_state):
            rng.set_state(rng_state.to(rng.device))

        if dont_have(seq_chunked):  # in this case, the sequence was too short for even a single chunk
            retrieved = None
        else:
            retrieved = torch.randint(
                self.vocab_size,
                (seq_chunked.shape[1],
                 self.chunk_size + self.continuation_length, self.num_neighbors,
                 *seq_chunked.shape[2:]),
                device=seq.device,
                generator=rng
            )  # [L//C, C', N, ...]
            retrieved = retrieved.movedim(0, 2)  # [C', N, L//C, ...]

        self.update_cache(last_partial_chunk, 'last_partial_chunk')
        self.update_cache(rng.get_state(), 'rng_state')

        return retrieved

    def __call__(self, *args, **kwargs):
        return self.retrieve(*args, **kwargs)


class Encoder(nn.Module, ModuleWithCache):
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
        ModuleWithCache.__init__(self)

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
        self.for_init = [m for m in chain(self.sa, self.ca, self.ff) if have(m)]

    def forward(self, y, x):
        """
        y: [C', N, L//C, ..., D']
        x: [L, ..., D]
        """
        last_partial_x_chunk = self.get_from_cache('last_partial_x_chunk')
        if have(last_partial_x_chunk):
            x = torch.cat((last_partial_x_chunk.to(x.device), x))
        x_chunked, last_partial_x_chunk = chunkenize(x, self.chunk_size)  # [C, L//C, ..., D]
        self.update_cache(last_partial_x_chunk, 'last_partial_x_chunk')
        if dont_have(y):
            return None
        else:
            for sa, ca, ff in zip(self.sa, self.ca, self.ff):
                y = sa(y)  # [C', N, L//C, ..., D']
                if have(ca):
                    y = ca(y, x_chunked)  # [C', N, L//C, ..., D']
                y = ff(y)  # [C', N, L//C, ..., D']
            return self.norm(y)  # [C', N, L//C, ..., D']


class Decoder(nn.Module, ModuleWithCache):
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
        ModuleWithCache.__init__(self)

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

        def use_cross(i):
            return (i >= ca_start) and ((i - ca_start) % ca_freq == 0)

        # self-attention blocks
        self.sa = nn.ModuleList(
            [AttentionBlock(d, num_heads=num_heads, causal=True, dropout=dropout, use_kv_cache=True)
             for _ in range(num_layers)])

        # cross-attention blocks
        self.ca = nn.ModuleList(
            [ChunkedCrossAttentionBlock(chunk_size, d, d_enc, num_heads=num_heads, dropout=dropout)
             if use_cross(i) else None for i in range(num_layers)])

        # feed-forward blocks
        self.ff = nn.ModuleList([FeedForwardBlock(d, d_ff, dropout=dropout)
                                 for _ in range(num_layers)])

        # normalization
        self.for_enc_norm = RMSNorm(d)
        self.norm = RMSNorm(d)

        self.chunk_size = chunk_size

        # we will use this for weight initialization of the final projection of each
        # attention/ff layer
        self.for_init = [m for m in chain(self.sa, self.ca, self.ff) if have(m)]

        self.register_modules_with_cache(list(self.sa + self.ca))

    def forward(self, x, y):
        # x: [L, ..., D]
        # y: [C', N, L//C, ..., D']
        z = None
        encoded = False

        for sa, ca, ff in zip(self.sa, self.ca, self.ff):
            x = sa(x)  # [L, ..., D]
            if have(ca):
                if not encoded:
                    z = self.encoder(y, self.for_enc_norm(x))  # [C', N, L//C, B, D']
                    encoded = True
                x = ca(x, z)  # [L, ..., D]
            x = ff(x)  # [L, ..., D]

        return self.norm(x)  # [L, ..., D]


class RetroLanguageModel(nn.Module, ModuleWithCache):
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
            ca_layers_enc,
            chunk_size,
            continuation_length,
            num_neighbors,
            dropout,
            share_embeddings=False
    ):
        nn.Module.__init__(self)
        ModuleWithCache.__init__(self)

        assert (not share_embeddings) or (d_enc == d_dec)

        self.retriever = Retriever(vocab_size, chunk_size, continuation_length, num_neighbors)
        self.embedding_dec = nn.Embedding(vocab_size, d_dec)
        self.embedding_enc = nn.Embedding(vocab_size, d_enc) if not share_embeddings \
            else partial(self.embedding_dec)
        self.encoder = Encoder(num_layers_enc, d_enc, d_ff_enc, d_dec, num_heads_enc,
                               ca_layers_enc, chunk_size, dropout)
        encoder = partial(self.encoder)
        self.decoder = Decoder(num_layers_dec, d_dec, d_ff_dec, d_enc, num_heads_dec,
                               ca_start_dec, ca_freq_dec, chunk_size, encoder, dropout)
        self.to_vocab = nn.Linear(d_dec, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.register_modules_with_cache([self.retriever, self.decoder, self.encoder])

        self.chunk_size = chunk_size

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
                assert len(list(module.parameters())) == 4  # to_q, to_kv, to_o, for_pos_enc
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
                module.to_out[0].data /= sqrt(len(subnet.for_init))

    def forward(self, seq, labels=None):

        # retrieve neighbors from database, for each chunk in `seq`
        retrieved = self.retriever(seq)  # [C', N, L//C, ...]

        # convert tokens to embeddings
        x = self.dropout(self.embedding_dec(seq))  # [L, ..., D]
        if have(retrieved):
            y = self.dropout(self.embedding_enc(retrieved))  # [C', N, L//C, ..., D']
        else:
            y = None

        # apply decoder and language model head
        dec_out = self.decoder(x, y)  # [L, ..., D]

        logits = self.to_vocab(dec_out)  # [L, ..., V]

        # calculate and return loss if labels were given
        loss = None
        if have(labels):
            loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss
