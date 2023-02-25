import torch
import torch.nn as nn
import math
from utils import clones
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
    Parameters:
        query: torch.tensor of size (N, Lq, d_k)
            where N = batch size, Lq = sequence length
        key: torch.tensor of size (N, Lk, d_k)
        value: torch.tensor of size (N, Lk, d_v)
        mask (used in q1.3): None or torch.tensor of size (N, Lk)
        dropout (used in q1.3): None or nn.Dropout()
        
    Returns:
        attn_out: Output, same size as value
        attn_weights: torch.tensor of size (N, Lq, Lk)
    
    """
    dk = query.size(dim=2)
    sqrt_dim = math.sqrt(dk)
    score = torch.matmul(query, key.transpose(1, 2))/sqrt_dim
    if mask is not None:
        # mask = mask.unsqueeze(1)
        if len(mask.size()) == 2:
            mask = mask.unsqueeze(1)
        score = score.masked_fill(mask == 0, -float("inf"))
    attn_weights = torch.nn.functional.softmax(score, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    attn_out = torch.matmul(attn_weights, value)
    return attn_out, attn_weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k (since that is true in transformers)
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implement forward pass of multi-headed attention"
        """
        Parameters:
            query: torch.tensor of size (N, Lq, d_model)
                where N = batch size, Lq = sequence length
            key: torch.tensor of size (N, Lk, d_model)
            value: torch.tensor of size (N, Lk, d_model)
            mask: None or torch.tensor of size (N, Lk)
            
        Set variable value:
            self.attn to attention values: size (N, h, Lq, Lk)

        Returns:
            attn_out: Output, same size as value

        """
        
        # Make sure to apply a final linear transformation to the output (HINT: self.linears)
        # as defined in the transformers paper (https://arxiv.org/pdf/1706.03762.pdf)
        # Make sure to use the 'mask'
        N, l_q, d_model = query.size(dim=0), query.size(dim=1), query.size(dim=2)
        l_k, l_v = key.size(dim=1), value.size(dim=1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)
        q = self.linears[0](query).view(N, l_q, self.h, self.d_k)
        k = self.linears[1](key).view(N, l_k, self.h, self.d_k)
        v = self.linears[2](value).view(N, l_v, self.h, self.d_k)
        q = q.transpose(1, 2).contiguous().view(N*self.h, l_q, self.d_k)
        k = k.transpose(1, 2).contiguous().view(N * self.h, l_k, self.d_k)
        v = v.transpose(1, 2).contiguous().view(N * self.h, l_v, self.d_k)
        attn_out, attn_weights = attention(q, k, v, mask=mask, dropout=self.dropout)
        self.attn = attn_weights.view(N, self.h, l_q, l_k)
        # print(attn_out.size(), l_k, l_q, d_model, self.d_k)
        attn_out = attn_out.view(N, self.h, l_q, self.d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(N, l_q, d_model)
        attn_out = self.linears[3](attn_out)
        # print(value.size(), attn_out.size())
        return attn_out
        
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

    

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())    

    
