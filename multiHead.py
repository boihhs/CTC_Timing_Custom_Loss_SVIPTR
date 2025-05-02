'''
Written by Leo Benahaon on 02/01/2025
This is a PyTorch implementation of multi-head attention
The input is a (b, H, W, C or d_in)
The output is also (b, H, W, C or d_in)

I also added the MultiHeadBlock on 02/06/2025
This is a way to use the MHA in a nice way and chain it
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHead(nn.Module):
    def __init__(self, numberOfHeads, d_in, dropout_prob=0.1):
        super(MultiHead, self).__init__()
        self.h = numberOfHeads
        self.d_in = d_in

        # Sanity check
        assert d_in % numberOfHeads == 0, "d_in must be divisible by numberOfHeads"
        self.d_h = d_in // numberOfHeads
        
        # Weight matrixes
        self.W_QKV = nn.Linear(d_in, d_in * 3, bias=False)
        self.W_O = nn.Linear(d_in, d_in, bias=False)

        # Dropout layer to apply on the attention weights
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xin):
        b, H, W, C = xin.shape  # C should equal d_in
        N = H*W
        xin = xin.reshape(b, N, self.d_in)
        # Linear projection followed by reshape into (b, 3, h, N, d_h)
        QKV = self.W_QKV(xin).reshape(b, 3, self.h, N, self.d_h)

        # Extract Q, K, V each with shape (b, h, N, d_h)
        Q = QKV[:, 0, :, :, :]
        K = QKV[:, 1, :, :, :]
        V = QKV[:, 2, :, :, :]
        
        # Compute scaled dot-product attention
        # Q @ K^T gives shape (b, h, N, N) -> (b, h, N, d_h) @ (b, h, d_h, N)
        scores = (Q @ K.transpose(-2, -1)) * (self.d_h ** -0.5)

        # Softmax over the last dimension
        attn = F.softmax(scores, dim=-1)

        # Apply dropout to the attention probabilities
        # This makes it so the model can't relay on a single relationship
        attn = self.dropout(attn)

        # Multiply the attention scores with V to get (b, h, N, d_h)
        out = attn @ V
        
        # Permute to (b, N, h, d_h) as you need to have h and d_h next to each other then reshape to (b, N, d_in)
        out = out.transpose(1, 2).reshape(b, N, self.d_in)
        
        # Final linear projection to connect all the heads together
        x_out = self.W_O(out)

        return x_out.reshape(b, H, W, self.d_in) # Returns in image format to say consistant
    

class MultiHeadBlock(nn.Module):
    def __init__(self, numberOfHeads, d_in, dropout_prob=0.1):
        super(MultiHeadBlock, self).__init__()

        self.d_in = d_in

        self.layer_norm1 = nn.LayerNorm(d_in)
        self.layer_norm2 = nn.LayerNorm(d_in)

        self.MHA = MultiHead(numberOfHeads, d_in)

        self.fc1 = nn.Linear(d_in, d_in*4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_in*4, d_in)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xin):
        # Xin is in shape (b, H, W, d_in)
        b, H, W, C = xin.shape

        xin = self.MHA(self.layer_norm1(xin)) + xin # Does the LN, MHA, and resdual connection

        xin = self.dropout(self.fc2(self.relu(self.fc1(self.layer_norm2(xin))))) + xin # Does the LN, FFN, dropout, and resdual connection

        # Returns (b, H, W, d_in)
        return xin