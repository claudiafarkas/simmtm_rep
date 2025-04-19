# USAGE IN PAPER DESCRIPTION:
# The decoder is part of the reconstruciton process. Helps map the aggregated point-wise representation 
# back to the original time series.

# Paper's Equation Interpretation Steps (see pg.4 "Point-wise Aggregation"):
# 1. The decoder taked in the aggregated point-wise representation i, 
# 2. Final reconstruciton of the orignal time series (output) is L * C
#    where L is the lenght of the time series and C is the number of channels
# 3. Uses an MLP at each time step.
# --------------------------------------------------------------------------------------------------------------------------
import torch

def decoder(input_dim, output_dim, hidden_dim = None):
    """
    Builds the shared decoder MLP.
    
    Args:
        input_dim: dimension of encoder output, d
        hidden_dim: hidden layer size, default = input_dim
        output_dim: output dimension of projector, default = input_dim
    """
    hidden_dim = hidden_dim or input_dim

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )
