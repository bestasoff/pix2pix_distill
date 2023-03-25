import torch
from torch import nn

class KernelAlignmentLoss(nn.Module):
    def forward(self, x_t, x_s):
        x_t_ = x_t.view(x_t.size(0), -1)
        x_s_ = x_s.view(x_s.size(0), -1)

        x_t_vec = (x_t_ * x_t_).sum(dim=1)
        x_s_vec = (x_s_ * x_s_).sum(dim=1)

        ret = (x_t_vec * x_s_vec).sum() / ((x_t_vec ** 2).sum() ** 0.5 * (x_s_vec ** 2).sum() ** 0.5)
        return -ret

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.register_buffer('zero_tensor', torch.tensor(0.))
    
    def get_zero_tensor(self, prediction):
        return self.zero_tensor.expand_as(prediction)
    
    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True
    ):
        if for_discriminator:
            if target_is_real:
                minval = torch.min(prediction - 1, self.get_zero_tensor(prediction))
                loss = -torch.mean(minval)
            else:
                minval = torch.min(-prediction - 1, self.get_zero_tensor(prediction))
                loss = -torch.mean(minval)
        else:
            loss = -torch.mean(prediction)
        
        return loss