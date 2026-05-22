import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, vector_dim, block_size, is_causal=False):
        super().__init__()
        self.vector_dim = vector_dim
        self.is_causal = is_causal
        self.wq = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wk = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wv = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wo = nn.Linear(vector_dim, vector_dim, bias=False)
        if self.is_causal:
            # The 'causal mask' is a lower-left triangular matrix of 1s, wrapped in
            # the outermost dimension(s) (which is the batch and, in the multihead
            # case, the number_of_heads dimension)
            causal_mask = torch.tril(torch.ones(block_size, block_size)).unsqueeze(0)
            # The next line creates a buffer 'self.mask'. Using a buffer rather than
            # a parameter ensures it will be moved to the GPU along with parameters,
            # but it won't be changed during training.
            self.register_buffer("mask", causal_mask)


    def compute_attention(self, q, k, v):
        # Shape of the tensors are (B,S,D) = (batch size, seq length, attention dim)
        # In the single-head case, attention_dim = vector_dim

        # YOUR CODE HER
        scores = torch.matmul(q, k.transpose(-2, -1)) #(B, S, S)
        scores = scores / math.sqrt(self.vector_dim)

        if self.is_causal:
            S = scores.shape[-1]
            scores = scores.masked_fill(self.mask[:, :S, :S] == 0, float('-inf'))

        # YOUR CODE HERE
        attention = scores.softmax(dim=-1) # (B, S, S)

        values = torch.matmul(attention, v) # (B, S, D)

        return values


    
    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        values = self.compute_attention(q, k, v)
        out = self.wo(values)
        return out



class MultiHeadSelfAttention(SelfAttention):
    def __init__(self, vector_dim, n_heads, block_size, is_causal=False):
        super().__init__(vector_dim, block_size, is_causal)
        self.att_dim = vector_dim//n_heads
        self.n_heads = n_heads


    def reshape_for_multihead_attention(self, x):
        """
        x has the shape (batch_size, seq_length, vector_dim)

        We want to split the representation of each token into 'number_of_heads'
        parts and treat each part separately. Thus, we need the returned tensor
        to have shape (batch_size, no_of_heads, seq_length, att_dim)
        """

        # YOUR CODE HERE
        B, S, D = x.shape
        x = x.reshape(B, S, self.n_heads, self.att_dim)
        x = x.transpose(1,2)


        return x


    def reshape_after_multihead_attention(self, x):
        """
        x has the shape (batch_size, no_of_heads, seq_length, att_dim)

        For each token, we now want to bring together the representation coming
        from each head. The returned token should have the shape:
        (batch_size, seq_length, vector_dim)
        """
        B, H, S, A = x.shape
        x = x.transpose(1,2)
        x = x.reshape(B, S, H * A)

        return x



    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.reshape_for_multihead_attention(q)
        k = self.reshape_for_multihead_attention(k)
        v = self.reshape_for_multihead_attention(v)
        values = self.compute_attention(q, k, v)
        values = self.reshape_after_multihead_attention(values)
        out = self.wo(values)
        return out
