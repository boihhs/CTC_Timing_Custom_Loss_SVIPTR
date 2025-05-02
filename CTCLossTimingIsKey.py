"""
***Custom CTC Loss***
The timing is key (along with some punishment... maybe)
Written by Leo Benaharon
"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    __constants__ = ['T']

    def __init__(self, T: int):
        super(CTCLoss, self).__init__()
        self.T = T
        
    # model_Out is B x T x N (N = number of classes)
    # LPB is a B x U'max containing the target sequences all padded to be the same length
    # UPB is the actual U' for each target sequence in the batch and is size B
    def forward(self, model_Out: torch.Tensor, LPB: torch.Tensor, UPB: torch.Tensor) -> torch.Tensor:
        
        device = model_Out.device
        LPB.to(device)
        UPB.to(device)
        B, T, N = model_Out.shape
        _, UP_Max = LPB.shape

        assert model_Out.size(1) == self.T, "model_Out's time dimension does not match T"


        """
        Generate the distrubation of tokens based on the normailzed paskels triangle.
        It will end up being size (B x T x UP_max) giving the model a hint on when to output certain values.
        This works as in theory in the image, the words will be centered in a way where this disturabtioin holds true.
        """
    
        t_indices = torch.arange(T, dtype=torch.float32).unsqueeze(0).to(device)  # Generate a vection lenght 24 which is [[0, 1, 2, 3, ..., 23]]
        shifts_float = - (UPB // 2).to(device).unsqueeze(1) + (((UPB-1)).unsqueeze(1)) * (t_indices / 24) # Now create the shifts based on the function y = -UPB/2 + (UPB-1)*(t/24)
        # What the above does is tell us how much we should shift the pascel part given the time and which number it is in the batch
        # On the right size you have (B x 1) + (B x 1)(1 x T) = (B x 1) + (B x T) = (B x T)
        shifts = shifts_float.floor().to(device)  # Floors it out and as we see above, this gives us a (B x T) tensor which shows us how much shift to apply
        # at a certian Time in the batch

        k = torch.arange(UP_Max, dtype=torch.float64).unsqueeze(0).to(device)  # Generates a vector length UP_max which is (after the unsqueeze) [[0, ..., UP_max]]
        n = UPB.unsqueeze(1) # n is now a (B x 1)
        log_coef = (torch.lgamma(n + 1) - (torch.lgamma(k + 1) + torch.lgamma(n - k + 1))).to(device) # Uses the formula (n k) = n!/(k!(n-k)!), where n is the row and k is the elemnet in the row
        # Remeber that !n = gammaFunction(n+1), so along that it is in log form the division turns to subtraction and 
        # above just makes a (B x 1) - ((1 x UP_max) + (B x UP_max)) = (B x UP_max)
        log_softmax_pascal = torch.log_softmax(log_coef, dim=1)  # Now normlize the get log problibties

       
        expanded_pascal = log_softmax_pascal.unsqueeze(1).expand(-1, T, -1)  # Unsqueeze takes it to (B x 1 x UP_max), then it makes it (B x T x UP_max)
        base_indices = torch.arange(UP_Max, device=device).unsqueeze(0).unsqueeze(0).to(device)  # Makes (1 x 1 x UP_max)
        # This below computes the indices of the rolling
        new_indices = ((base_indices - shifts.unsqueeze(-1)) % UP_Max).to(dtype=torch.int64)  # Does (1 x 1 x UP_max) - (B x T x 1) = (B x T x UP_max)
        # Make a tensor of the rolled pascels triangle using the indices of the last dimention of new_indices to get the values of the last dimention of expanded_pascal to get the new matrix
        rolling = torch.gather(expanded_pascal, dim=2, index=new_indices).to(device)  # This is a size of (B x T x UP_max)

        col_idx = torch.arange(UP_Max, device=device).unsqueeze(0).unsqueeze(0)  # Creates a matrix size (1 x 1 x UP_max) which can be seen as the column indicies of the rolling
        s_expanded = shifts.unsqueeze(-1)  # Makes it into (B x T x 1)
        # Now create the mask
        mask_wrapped = ((s_expanded > 0) & (col_idx < s_expanded)) | ((s_expanded < 0) & (col_idx >= UP_Max + s_expanded))
        # Above basicly says that if the shift is too much where it goes back to the start set it to true or if it goes so back it goes to the end, set it to true
        rolled_result = rolling.clone() # Clone it 
        rolled_result[mask_wrapped] = -1e9 # Thus if did a loopty loop set it near -inf

        """
        Generate Mask Z which is B x T x UP_Max
        ln(Z(B, t, u)) = 0 if u >= UPB - 2(T - t)
        and if not is -infinity
        """

        UP_expand = UPB.view(B, 1, 1).float() # UP_expand[B, 1, 1] = UPB[B] -> UP_expand = [[UPB[0]];...; [UPB[B]]]
        t_range = torch.arange(0, T, device=device).view(1, T, 1) # t_expand = [[[0]; [1]; [2]; ...; [T]]]
        u_range = torch.arange(0, UP_Max, device=device).view(1, 1, UP_Max) # u_expand = [[0, 1, 2, ..., UP_Max]]
        
        threshold = UP_expand.to(device) - 2 * (T - t_range)

        '''
        (UP_expand - 2 * (T - t_expand)) is B x T x 1 = (B x T x 1) - 2 * ((B x T x 1) - (B x T x 1)) =
        [UPB[0],..., UPB[0];            [T, ..., T;         [0, 1, 2, ..., T;
         UPB[1],..., UPB[1];             T, ..., T;          0, 1, 2, ..., T;
                :               - 2 * (      :         -            :
                :                            :                      :
         UPB[B],..., UPB[B]]              T, ..., T]         0, 1, 2, ..., T]

         This alligns with our formula as we wan't to keep UPB constant as we change time.
         Also I remove the last dimention in this as it is just 1 so, yeah...
        '''

        valid = u_range >= threshold

        '''
        This expands u_expand from (1 x 1 x UP_Max) into (B x T x UP_Max).
        Threshold is also expanded from (B x T x 1) into (B x T x UP_Max).
        This alligns with our inequality compares each u with each batch's time.
        This is seen as the threshold (B x T) is repeated UP_Max times and is compared with u_expand.
        '''

        lnZ = torch.where(valid.to(device), model_Out.new_full((), 0.0), model_Out.new_full((), -1e9)) # Get lnZ where if valid = 1, set to 0, else set to -inf

        """
        Generate Transistion matrix M which is (B x UP_Max x UP_Max)
        ln(M(B, u, i)) = 0 if f(B, u) <= i <= u, -inf otherwise
        where
        f(B, u) = u - 1 if LPB[u] = blank token or LPB[u] = LPB[u-2];
        -inf otherwise
        """

        i_range = torch.arange(0, UP_Max, device=device).view(1, 1, UP_Max) # i_expand = [[0, 1, 2, ..., UP_Max]]
        u_range = torch.arange(0, UP_Max, device=device).view(1, UP_Max, 1) # u_expand = [[0], [1], [2], ..., [UP_Max]]]

        '''
        The goal of the below is the create the f(B, u).
        So we need to check if the condition of LPB[u] = blank token or LPB[u] = LPB[u-2].
        This is equivlent to LPB[u] = LPB[u-2] if we define all LPB[-number] = 0.
        However, this is not true. But we can just shift everything down and see if LPB[:, u] = LPB[:, u-2].
        But this is not suffecent as the first non blank token needs to be u-1. Therefore I can extend
        the LPB and add the LPB[:, 1] before the LPB and thus the LPB_extended[:, 0] = LPB_extended[:, 2],
        solving our problem
        '''
        LPB_extend = torch.cat([LPB[:, 1].unsqueeze(1), LPB], dim = 1)
        LPB_extend_shift = torch.zeros_like(LPB_extend)
        LPB_extend_shift[:, 2:] = LPB_extend[:, :-2] # Shift everything 2 down

        f_valid = LPB_extend_shift == LPB_extend # True if we should do u - 1, False if u - 2
        f_valid = f_valid[:, 1:] # Remove the extended column
        f = torch.full_like(f_valid, 0, dtype=torch.long) # Need to set it as a long instead of a boolean, shape of (B x UP_Max)

        u_expanded = u_range.expand(B, UP_Max, 1).squeeze(-1)  # Match shape (B x UP_Max) so can use f_valid on it
        # Apply rule for f
        f = torch.where(f_valid.to(device), u_expanded - 1, u_expanded - 2).long() # Need to .long to convert it from a boolean to a long
        f.clamp_(min=0) # Need to clamp the first column as it always is -1

        # Apply the unsqueeze(2) to the f as we want f to be in shape (2 x 5 x 1) so that the (2 x 5) part is compared to each "i" independently 
        M_valid = (f.unsqueeze(2) <= i_range) & (i_range <= u_range) # Generate mask, shape of (B x UP_Max x UP_Max)

        lnM = torch.where(M_valid.to(device), model_Out.new_full((), 0.0), model_Out.new_full((), -1e9)).to(device)

        """
        Now lets creat the lny (log_probs). This is size (B x T x N)
        This is ln(y) = x - c - ln(sum{e^{x-c}}) where c is the max of x
        """
        # Generate the punishment for the blanks
        punish = torch.zeros(N)
        punish[0] = torch.log(torch.tensor([.65]))
        punish = punish.unsqueeze(0).unsqueeze(0) # Now is (1 x 1 x N)

        # Use keepdim=True as we don't want to have our N dimention to be removed as it can't broadcast (BxT) into (BxTxN)
        # But can automatically broadcast (B x T x 1) into (B x T x N)
        c, _ = torch.max(model_Out, 2, keepdim=True)
        logSum_Model_Out = torch.log(torch.sum(torch.exp(model_Out - c), 2, keepdim=True))
        lny = model_Out - c - logSum_Model_Out
       
        
        # But we need to make a matrix of size (B x T x UP_Max). As LPB is (B x U'max) we make it into (B x 1 x UP_Max)
        # Then expand it into (B x T x UP_Max). Then as lny is  (B x T x N), what the gather does
        # is that it returns a array of (B x T x UP_Max) where the B and T stay the same, but it replaces each value of
        # the l'_u in the UP_Max with its proablity by indexing into the lny
        lny_target = lny.gather(2, LPB.unsqueeze(1).expand(B, T, UP_Max).to(device)) + rolled_result

        """
        Create ln(a) with initial conditions (we will not be using the T dimention as we will be saving memory.
        Plus it is unnessary to keep it if you aren't debuging).
        lna is size (B x UP_Max)
        """

        lna = torch.full((B, UP_Max), -1e9).to(device)
        lna[:, 0] = lny_target[:, 0, 0]
        lna[:, 1] = lny_target[:, 0, 1]

        """
        Now create the for loop for the time and update lna until we reach t = T
        """
    
        for t in range(1, T):
            # Add mask matrix (M) with the lna to generate which u (out of UP_max) is added together to make the next lna
            log_sum_accum = lnM + lna.unsqueeze(1) # Unsqueeze lna from (B x UP_Max) into (B x UP_Max x 1) so it can be broadcasted into (B x UP_Max x UP_Max) automaticlly
            
            # Get the max, c which should be shape (B x UP_Max x 1) so it can broadcast
            c, _ = torch.max(log_sum_accum, 2, keepdim=True)
           
            log_sum = torch.log(torch.sum(torch.exp((log_sum_accum - c).nan_to_num(0.0)), 2, keepdim=True))
            # Finally get the updated lna. The squeeze converts everything into (B x UP_Max)
            lna = lny_target[:, t, :] + c.squeeze(-1) + log_sum.squeeze(-1) + lnZ[:, t, :]

        """
        Compute the final lnp which is of shape (B)
        """
        # UPB is made into a (B x 1) and then for each value in UPB, it then extracts that index from the batch of lna
        # This returns a (B x 1) which then is squeezed back into (B)
        lnp_val1 = lna.gather(1, (UPB - 1).unsqueeze(1).to(device)).squeeze(1).to(device)
        lnp_val2 = lna.gather(1, (UPB - 2).unsqueeze(1).to(device)).squeeze(1).to(device)
        c = torch.max(lnp_val1, lnp_val2)
        lnp_sum = torch.log(torch.exp(lnp_val1 - c) + torch.exp(lnp_val2 - c))
        lnp = c + lnp_sum

        return -lnp.mean()

        
        
       
















        