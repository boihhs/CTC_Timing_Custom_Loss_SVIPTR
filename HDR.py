'''
Written by Leo Benahaon on 02/08/2025
This is an implementation of thhe height dimentsion reduction (HDR). The goal of this is to increase dimention size while decreasing height size.
This makes the model more effecent as it has to compress information.
In the final stage it averages over the H dimention and and increases the dimention before going to the CTC decoder.
This makes it so that there will be 96/4 outputs from the CTC decoder.
The input is (b, H, W, d_in/C) and the output is (b, H, W, d_out) if it isn't the final stage and
is (b, W, d_out) if it is the final stage.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class HDR(nn.Module):
    def __init__(self, d_in, d_out, finalStage = False, dropout_prob=0.1):
        super(HDR, self).__init__()

        self.d_in = d_in

        self.d_out = d_out

        self.finalStage = finalStage

        # Is it the final stage?
        if self.finalStage:
            # If so do this
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
            

            self.fc = nn.Linear(self.d_in, self.d_out) 
            
            self.activation = nn.Hardswish()
            

            self.dropout = nn.Dropout(dropout_prob)
        
        else:
            # Else, just compress the H into H/2
            self.conv = nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=3, padding=1, stride=(2,1))

            self.layer_norm = nn.LayerNorm(self.d_out)


    def forward(self, xin):
        # Xin is in shape (b, H, W, d_in)
        b, H, W, C = xin.shape
        # Make into a conv2d input (b, d_in, H, W)
        xin = xin.permute(0, 3, 1, 2)

        if self.finalStage:

            xin = self.adaptive_pool(xin) # Avereges over the H dimention to make (d, d_in, 1, W)

            xin = xin.squeeze(2) # Removes the H dimention to make (b, d_in, W)

            xin = xin.permute(0, 2, 1) # (Go back to the nice way: b, W, d_in)

            xin = self.dropout(self.activation(self.fc(xin))) # Increase the dimention size while providing nonlineartiy to make (b, W, d_out)

            return xin # (b, W, d_out)
        else:
            xin = self.conv(xin) # Halves the H dimention and increases channell dimention to make (b, d_out, H/2, W)

            return self.layer_norm(xin.permute(0, 2, 3, 1)) # Makes it in the nice form and then layer norms over the channels: (b, H/2, W, d_out)
   
