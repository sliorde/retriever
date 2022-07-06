# TODO:
#  * retrieval preprocessing+lookup
#  * make transformer model more flexible so that it's easy to retrofit different given models
#  * cached inference
#  * share caches of masks and position encoding between different layers
#  * training loop, parallelization
from math import sqrt
from functools import lru_cache, partial
from itertools import chain

import torch
import torch.nn as nn
from torch.nn.functional import pad, softmax
import opt_einsum


def view_or_reshape(x, *tensor_or_shape):
    """
    change the view of tensor x to the specified shape or to the specified tensor's shape; if impossible, reshape
    (`reshape` incurs a copy whereas `view` does not)
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
    For example, if `x` is a tensor with shape`[24,1,2,3,4,5]` representing a sequence of length 24 and we use `
    `dim=0` and `chunk_size=4`, we will get a tensor with shape `[4,6,1,2,3,4,5]`, where the `6` is in the chunk
    index dimension, and `4` is in the within-chunk dimension.
    TODO: maybe we should change the ordering of dimensions of all objects so that we don't need to do transpose
     here. The transpose operation incurs a copy later.
    """
    dim = dim % x.ndim
    return view_or_reshape(x, *x.shape[:dim], -1, chunk_size, *x.shape[(dim + 1):]).transpose(dim, dim + 1)


def unchunk(x, dim=0):
    """
    the inverse of `chunk()`
    """
    return view_or_reshape(x.transpose(dim, dim + 1), *x.shape[:dim], -1, *x.shape[(dim + 2):])


@lru_cache(128)
def cached_einsum_path(subscripts, *shapes):
    """
    we use an einsum function that optimizes the reduction order of indices in a tensor   
    """
    return opt_einsum.contract_expression(subscripts, *shapes)


def einsum(subscripts, *operands):
    path = cached_einsum_path(subscripts, *[tuple(o.shape) for o in operands])
    return path(*operands)


def attention(x_q, x_kv, to_q, to_k, to_v, to_out, mask=None, positional_encoding=None, flatten_dims=None,
              dropout=None):
    d = x_q.size(-1)
    if positional_encoding is None:
        logits = einsum('dhb,n...b,dhc,m...c->nm...h', to_q, x_q, to_k, x_kv)
    else:
        # ilke transformerXL, but without bias terms
        q = einsum('dhb,n...b->nd...h', to_q, x_q)
        content_logits = einsum('nd...h,dhc,m...c->nm...h', q, to_k, x_kv)
        position_logits = einsum('nd...h,nmdh->nm...h', q, positional_encoding)
        dim_diff = position_logits.ndim - content_logits.ndim
        if dim_diff > 0:
            content_logits = view_or_reshape(content_logits, *content_logits.shape[:2], *(1,) * dim_diff,
                                             *content_logits.shape[2:])
        elif dim_diff < 0:
            position_logits = view_or_reshape(position_logits, *position_logits.shape[:2], *(1,) * (-dim_diff),
                                              *position_logits.shape[2:])
        logits = content_logits + position_logits
    logits /= sqrt(d)

    if mask is not None:
        logits.masked_fill_(mask[(slice(None), slice(None), *(None,) * (logits.ndim - 2))],
                            float('-inf'))

    if flatten_dims is not None:
        logits = view_or_reshape(logits, logits.size(0), -1, *logits.shape[flatten_dims + 1:])
        x_kv = view_or_reshape(x_kv, -1, *x_kv.shape[flatten_dims:])
    weights = softmax(logits, 1)
    if dropout is not None:
        weights = dropout(weights)
    out = einsum('dch,chb,m...b,nm...h->n...d', to_out, to_v, x_kv, weights)
    return out


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        nn.Module.__init__(self)

        self.eps = eps
        self.dim_norm = d ** (-1. / 2)

        self.scale = nn.Parameter(torch.ones(d, **factory_kwargs))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) * self.dim_norm
        normed = x / (rms + self.eps)
        scaled = self.scale * normed
        return scaled


class Attention(nn.Module):
    def __init__(self, d, d_x_q=None, d_x_kv=None, d_qk=None, d_v=None, num_heads=1, causal=False, offset=0,
                 flatten_dims=None, dropout=0.0):
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

        self.mask_cache = dict()
        self.pos_enc_cache = dict()

        self.d_x_q = d_x_q
        self.d_x_kv = d_x_kv
        self.d_qk = d_qk
        self.d_v = d_v
        self.num_heads = num_heads
        self.offset = offset or 0
        self.causal = causal
        self.flatten_dims = flatten_dims

    def get_causal_mask(self, sz, device):
        if sz in self.mask_cache:
            mask = self.mask_cache[sz].to(device)
        else:
            mask = torch.ones(sz, sz, device=device, dtype=torch.bool).triu(1)
            self.mask_cache[sz] = mask
            if len(self.mask_cache) > 10:
                self.mask_cache.pop(next(iter(self.mask_cache)))
        return mask

    def get_pos_enc(self, sz_q, sz_k, offset, device):
        # TODO: for training, we cache the position encoding before applying the linear transform,
        #  because the transform is trainable and changes between batches...but for eval, we might
        #  as well cache the result of the transform.
        #
        # TODO: we use relative encoding of TransformerXL, but without the bias terms. This is what
        #  I think is implied by the retro paper, but it's not defined explicitly.
        key = (sz_q, sz_k, offset)
        if key in self.pos_enc_cache:
            sincos, dists = self.pos_enc_cache[key]
            sincos = sincos.to(device)
            dists = dists.to(device)
        else:
            dists = torch.arange(sz_q, device=device)[:, None] - \
                    torch.arange(sz_k, device=device)[None, :] + offset  # [sz_q, sz_k]
            r = torch.arange(dists.min(), dists.max() + 1.0, 1.0, device=device)  # [sz_q+sz_k-1]
            inv_freq = 1 / (10000 ** (torch.arange(0.0, self.d_x_kv, 2.0, device=device) / self.d_x_kv))  # [d_x_kv/2]
            phases = r[:, None] * (inv_freq[None, :])  # [sz_q+sz_k-1, d_x_kv/2]
            sincos = torch.cat([phases.sin(), phases.cos()], dim=-1)  # [sz_q+sz_k-1, d_x_kv]

            self.pos_enc_cache[key] = (sincos, dists)
            if len(self.pos_enc_cache) > 10:
                self.pos_enc_cache.pop(next(iter(self.pos_enc_cache)))
        pos_enc = einsum('dhb,nb->ndh', self.for_pos_enc, self.dropout(sincos))  # [sz_q+sz_k-1, d_qk, h]
        pos_enc = pos_enc[dists - dists.min()]  # [sz_q, sz_k, d_qk, h]
        return pos_enc

    def forward(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
        sz = x_q.size(0)
        device = x_q.device
        if self.causal:
            assert x_kv is x_q
            mask = self.get_causal_mask(sz, device)
        else:
            mask = None
        pos_enc = self.get_pos_enc(x_q.size(0), x_kv.size(0), self.offset, device)

        return attention(x_q, x_kv, self.to_q, self.to_k, self.to_v, self.to_out, mask, pos_enc, self.flatten_dims,
                         self.dropout)


class AttentionLayer(nn.Module):
    def __init__(self, d, d_kv=None, num_heads=1, causal=False, dropout=0.0):
        nn.Module.__init__(self)
        if d_kv is None:
            d_kv = d
        self.norm = RMSNorm(d)
        self.attention = Attention(d, d_x_kv=d_kv, num_heads=num_heads, causal=causal, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # we need this for initialization:
        self.to_out = self.attention.to_out

    def forward(self, x_q, x_kv=None):
        return x_q + self.dropout(self.attention(self.norm(x_q), x_kv))


class ChunkedCrossAttentionLayer(nn.Module):
    def __init__(self, chunk_size, d, d_kv, num_heads=1, dropout=0.0):
        nn.Module.__init__(self)
        if d_kv is None:
            d_kv = d
        self.norm = RMSNorm(d)
        self.attention = Attention(d, d_x_kv=d_kv, num_heads=num_heads, offset=chunk_size, flatten_dims=2)
        self.dropout = nn.Dropout(dropout)

        self.chunk_size = chunk_size

        # we need this for initialization:
        self.to_out = self.attention.to_out

    def forward(self, x_q, x_kv):
        x_q_chunked = chunkenize(
            pad(self.norm(x_q)[self.chunk_size - 1:], (0, 0) * (x_q.ndim - 1) + (0, self.chunk_size - 1)),
            self.chunk_size)  # [C, L//C, B, D]
        z = self.attention(x_q_chunked, x_kv)  # [C, L//C , B, D]
        z = pad(unchunk(z), (0, 0) * (x_q.ndim - 1) + (z.size(0) - 1, 0))[
            :x_q.size(0)]
        return x_q + self.dropout(z)


class FeedForwardLayer(nn.Module):
    def __init__(self, d, d_ff, dropout=0.0):
        nn.Module.__init__(self)
        self.ff = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
            nn.Dropout(dropout)
        )

        # we need this for initialization:
        self.to_out = [m for m in self.ff.modules() if isinstance(m, nn.Linear)][-1].weight

    def forward(self, x):
        return x + self.ff(x)


class Retriever:
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
        # seq [L, B]
        seq_chunked = chunkenize(seq, self.chunk_size)  # [C, L//C, B]   #in paper: =C

        retrieved = torch.randint(
            self.vocab_size,
            (seq_chunked.shape[1],
             self.chunk_size + self.continuation_length, self.num_neighbors,
             *seq_chunked.shape[2:]),
            device=seq.device,
            generator=torch.Generator(device=seq.device).manual_seed(29847892371)
        )
        return retrieved.movedim(0, 2)


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

        self.sa = nn.ModuleList([AttentionLayer(d, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.ca = nn.ModuleList(
            [AttentionLayer(d, d_dec, num_heads=num_heads, dropout=dropout) if i in ca_layers else None for i in
             range(num_layers)])
        self.ff = nn.ModuleList([FeedForwardLayer(d, d_ff, dropout=dropout) for _ in range(num_layers)])
        self.norm = RMSNorm(d)

        self.chunk_size = chunk_size

        # we will use this for initializing the final projection of each attention/ff layer
        self.for_init = [m for m in chain(self.sa, self.ca, self.ff) if m is not None]

    def forward(self, y, x):  # in paper: retrieved->RET(C),  y->H
        """
        y: [L, B, D]
        x: [C', N, L//C, B]
        """
        x = chunkenize(x, self.chunk_size)  # [C, L//C, B, D]
        for sa, ca, ff in zip(self.sa, self.ca, self.ff):  # in paper: i->p'
            y = sa(y)
            if ca is not None:
                y = ca(y, x)
            y = ff(y)
        return self.norm(y)


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

        self.sa = nn.ModuleList(
            [AttentionLayer(d, num_heads=num_heads, causal=True, dropout=dropout) for _ in range(num_layers)])
        self.ca = nn.ModuleList(
            [ChunkedCrossAttentionLayer(chunk_size, d, d_enc, num_heads=num_heads, dropout=dropout)
             if use_cross(i) else None
             for i in range(num_layers)])
        self.ff = nn.ModuleList([FeedForwardLayer(d, d_ff, dropout=dropout) for _ in range(num_layers)])
        self.norm = RMSNorm(d)

        self.chunk_size = chunk_size

        # we will use this for initializing the final projection of each attention/ff layer
        self.for_init = [m for m in chain(self.sa, self.ca, self.ff) if m is not None]

    def forward(self, x, y):
        encoder_invoked = False

        for sa, ca, ff in zip(self.sa, self.ca, self.ff):
            x = sa(x)
            if ca is not None:
                if not encoder_invoked:
                    z = self.encoder(y, x)  # [C', N, L//C, B, D]  in paper: =E
                    encoder_invoked = True
                x = ca(x, z)
            x = ff(x)
        return self.norm(x)


class RetroLanguageModel(nn.Module):
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
    ):
        nn.Module.__init__(self)
        self.retriever = Retriever(vocab_size, chunk_size, continuation_length, num_neighbors)
        self.embedding_enc = nn.Embedding(vocab_size, d_enc)
        self.embedding_dec = nn.Embedding(vocab_size, d_dec)
        self.encoder = Encoder(num_layers_enc, d_enc, d_ff_enc, d_dec, num_heads_enc, ca_layers, chunk_size,
                               dropout)
        encoder = partial(self.encoder)
        self.decoder = Decoder(num_layers_dec, d_dec, d_ff_dec, d_enc, num_heads_dec, ca_start_dec, ca_freq_dec,
                               chunk_size,
                               encoder, dropout)
        self.to_vocab = nn.Linear(d_dec, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        def xavier_(p):
            nn.init.xavier_normal_(p.view(-1, p.size(-1)), 1.0)

        def init(module):
            if isinstance(module, RMSNorm):
                assert len(list(module.parameters())) == 1  # scale
                nn.init.constant_(module.scale, 1.0)
            elif isinstance(module, Attention):
                assert len(list(module.parameters())) == 5  # to_q, to_k, to_v, to_out, for_pos_enc
                for p in module.parameters():
                    xavier_(p)
            elif isinstance(module, nn.Linear):
                assert len(list(module.parameters())) == 2  # weight, bias
                xavier_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                assert len(list(module.parameters())) == 1  # weight
                xavier_(module.weight)

        self.apply(init)

        # from GPT2 paper::
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        for subnet in [self.decoder, self.encoder]:
            for module in subnet.for_init:
                module.to_out.data /= sqrt(len(subnet.for_init))

    def forward(self, seq):
        retrieved = self.retriever.retrieve(seq)  # [C', N, L//C, B]   in paper: =RET(C)

        x = self.dropout(self.embedding_dec(seq))  # [L, B, D]  in paper: =H
        y = self.dropout(self.embedding_enc(retrieved))  # [C', N, (L//C), B, D]

        return self.to_vocab(self.decoder(x, y))


if __name__ == '__main__':

    torch.set_printoptions(precision=3, threshold=1000, linewidth=1000000)

    vocab_size = 1000
    seq_length = 24  # in paper: =n=2048
    batch_size = 3

    seq = torch.randint(vocab_size, (seq_length, batch_size))  # [L, B]  in paper: =X,

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
        ca_layers={1},  # in paper: =P_enc={1}
        chunk_size=4,  # in paper: =m=64  (this also equals the neighbor chunk size)
        continuation_length=4,  # in paper: =64
        num_neighbors=2,  # in paper: =k
        dropout=0.1,
    ).eval()

    out = model.forward(seq)

    print(seq.shape, flush=True)
    print(out.shape, flush=True)
