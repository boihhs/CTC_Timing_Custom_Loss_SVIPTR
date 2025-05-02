'''
Written by Leo Benahaon on 02/05/2025
This is a PyTorch implementation of multi-head attention for Decomposed Manhattan Self-Attention (MaSAd)
I belive that this is the most readable code avaible for this algo.
The input is a (b, H, W, C or d_in)
The output is also (b, H, W, C or d_in)

I also added the MultiHeadMaSAdBlock on 02/06/2025
This is a way to use the MHMaSA in a nice way and chain it
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadMaSAd(nn.Module):
    def __init__(self, numberOfHeads, d_in, dropout_prob=0.1):
        super(MultiHeadMaSAd, self).__init__()
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

    def generateDecay(self, l, device):
        # Returns (h, l, l)
        # This generates the decay matrix where it makes it so that tokens farther from each other are punished
        a = 2
        b = 6 
        index = torch.arange(l, device=device) # Need to use GPU when needed
        # Compute distance from each token from each in the attention
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()
        # Each head as a different scaling factor
        gamma = 1-2**(-a - ((b-a)*torch.arange(self.h, dtype=torch.float32)) / self.h) # Compute gamma from paper
        end = gamma[:, None, None].to(device)*mask.to(device) # Generate the h number of masks for the number of heads
        return end

    def forward(self, xin):
        b, H, W, C = xin.shape  # C should equal d_in

        # Linear projection into (b, H, W, 3*din) followed by reshape into (b, H, 3, h, W, d_h)
        QKV_W = self.W_QKV(xin).reshape(b, H, 3, self.h, W, self.d_h)

        # Generate (b, W, 3, h, H, d_h)
        QKV_H = QKV_W.transpose(1, 4)

        # Extract Q_W, K_W each with shape (b, H, h, W, d_h)
        Q_W = QKV_W[:, :, 0, :, :, :]
        K_W = QKV_W[:, :, 1, :, :, :]

        # Extract Q_H, K_H each with shape (b, W, h, H, d_h)
        Q_H = QKV_H[:, :, 0, :, :, :]
        K_H = QKV_H[:, :, 1, :, :, :]
        
        # Extract V with the shape (b, H, h, W, d_h)
        # Only need the V from the QKV_W
        V = QKV_W[:, :, 2, :, :, :]
        
        ## Compute scaled dot-product attention
        # Q_W @ K_W^T gives shape (b, H, h, W, W) -> (b, H, h, W, d_h) @ (b, H, h, d_h, W)
        scores_W = ((Q_W @ K_W.transpose(-2, -1)) * (self.d_h ** -0.5)) - self.generateDecay(W, xin.device)
        # Q_H @ K_H^T gives shape (b, W, h, H, H) -> (b, W, h, H, d_h) @ (b, W, h, d_h, H)
        scores_H = ((Q_H @ K_H.transpose(-2, -1)) * (self.d_h ** -0.5)) - self.generateDecay(H, xin.device)

        # Softmax over the last dimension
        attn_W = F.softmax(scores_W, dim=-1)
        attn_H = F.softmax(scores_H, dim=-1)

        # Apply dropout to the attention probabilities
        # This makes it so the model can't relay on a single relationship
        attn_W = self.dropout(attn_W)
        attn_H = self.dropout(attn_H)

        # Multiply the attention scores with V to get (b, H, h, W, d_h) -> ((b, W, h, H, H) @ ((b, H, h, W, W)@(b, H, h, W, d_h)).T).T
        out = (attn_H@((attn_W @ V).transpose(1, 3))).transpose(1, 3)
        
        # Permute to (b, H, W, h, d_h) as you need to have h and d_h next to each other then reshape to (b, H, W, d_in)
        out = out.transpose(2, 3).reshape(b, H, W, self.d_in)
        
        # Final linear projection to connect all the heads together
        x_out = self.W_O(out)

        return x_out


class MultiHeadMaSAdBlock(nn.Module):
    def __init__(self, numberOfHeads, d_in, dropout_prob=0.1):
        super(MultiHeadMaSAdBlock, self).__init__()

        self.d_in = d_in

        self.layer_norm1 = nn.LayerNorm(d_in)
        self.layer_norm2 = nn.LayerNorm(d_in)

        self.MHMaSA = MultiHeadMaSAd(numberOfHeads, d_in)

        self.fc1 = nn.Linear(d_in, d_in*4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_in*4, d_in)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, xin):
        # Xin is in shape (b, H, W, d_in)
        b, H, W, C = xin.shape

        xin = self.MHMaSA(self.layer_norm1(xin)) + xin # Does the LN, MHMaSA, and resdual connection

        xin = self.dropout(self.fc2(self.relu(self.fc1(self.layer_norm2(xin))))) + xin # Does the LN, FFN, dropout, and resdual connection

        # Returns (b, H, W, d_in)
        return xin