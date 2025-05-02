"""
This is the SVIPTR network which outputs the logged problilties of each class.
Written by Leo Benaharon on 03/02/2025
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from HDR import HDR
from MaSAd import MultiHeadMaSAdBlock
from multiHead import MultiHeadBlock
from ImageEmbed import PatchEmbedding
import numpy as np

class SVIPTR(nn.Module):
    def __init__(self):
        super(SVIPTR, self).__init__()

        # Patch Embedding
        self.patchEmbedding = PatchEmbedding()

        # First Local Attention
        self.MaSA_1_1 = MultiHeadMaSAdBlock(numberOfHeads=2, d_in=64)
        self.MaSA_1_2 = MultiHeadMaSAdBlock(numberOfHeads=2, d_in=64)
        self.MaSA_1_3 = MultiHeadMaSAdBlock(numberOfHeads=2, d_in=64)

        # First HDR
        self.HDR_1 = HDR(d_in=64, d_out=128)

        # posital embedding needed (optimal but improves preformece)

        # Second Local Attention
        self.MaSA_2_1 = MultiHeadMaSAdBlock(numberOfHeads=4, d_in=128)
        self.MaSA_2_2 = MultiHeadMaSAdBlock(numberOfHeads=4, d_in=128)
        self.MaSA_2_3 = MultiHeadMaSAdBlock(numberOfHeads=4, d_in=128)

        # First Global Attention
        self.MHSA_1_1 = MultiHeadBlock(numberOfHeads=4, d_in=128)
        self.MHSA_1_2 = MultiHeadBlock(numberOfHeads=4, d_in=128)
        self.MHSA_1_3 = MultiHeadBlock(numberOfHeads=4, d_in=128)

        # Second HDR
        self.HDR_2 = HDR(d_in=128, d_out=256)

        # Second Global Attention
        self.MHSA_2_1 = MultiHeadBlock(numberOfHeads=8, d_in=256)
        self.MHSA_2_2 = MultiHeadBlock(numberOfHeads=8, d_in=256)
        self.MHSA_2_3 = MultiHeadBlock(numberOfHeads=8, d_in=256)

        # Final HDR
        self.HDR_3 = HDR(d_in=256, d_out=192)

        self.fc_out_1 = nn.Linear(in_features=192, out_features=192*4)
        self.relu = nn.ReLU()
        # Replace 40 with number of possible text
        self.fc_out_2 = nn.Linear(in_features=192*4, out_features=50)


    def forward(self, xin):
        # The input for anything should be (B x H x W x d_in) where C = d_in
        # As such xin should be in this size

        xin = self.patchEmbedding(xin)

        # First Local Attention
        xin = self.MaSA_1_1(xin)
        xin = self.MaSA_1_2(xin)
        xin = self.MaSA_1_3(xin)

        # First HDR
        xin = self.HDR_1(xin)

        # posital embedding needed (optimal but improves preformece)

        # Second Local Attention
        xin = self.MaSA_2_1(xin)
        xin = self.MaSA_2_2(xin)
        xin = self.MaSA_2_3(xin)

        # First Global Attention
        xin = self.MHSA_1_1(xin)
        xin = self.MHSA_1_2(xin)
        xin = self.MHSA_1_3(xin)

        # Second HDR
        xin = self.HDR_2(xin)

        # Second Global Attention
        xin = self.MHSA_2_1(xin)
        xin = self.MHSA_2_2(xin)
        xin = self.MHSA_2_3(xin)

        # Final HDR
        xin = self.HDR_3(xin)

        xin = self.fc_out_1(xin)
        xin = self.relu(xin)
        # Replace 40 with number of possible text
        xin = self.fc_out_2(xin)

        '''
        The output of this should be (B x 1 x 24 x C) where C is the number of classes
        In the CTC documnation they want it to be in format (T x N x C),
        Where T is the input lenght, N is the batch size and C is the number of classes.
        This also should be log_softmaxed
        ''' 
        xin = torch.squeeze(xin, dim=1) # Now is (B x 24 x C)


        #xin = nn.functional.log_softmax(xin, dim=2)

        #xin = xin.permute(1, 0, 2) # Output in (24 x B x C)

        return xin

    

