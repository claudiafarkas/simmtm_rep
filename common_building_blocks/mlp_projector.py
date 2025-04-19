# USAGE IN PAPER DESCRIPTION:
# Is a shared MLP used to convert point-wise representations from the encoder to series-wise representations

# Paper's Equation Interpretation Steps (see pg.4 "Representation Learning):
## 1. the encoder gives point-wise representations; 1 vector / time step 
#     Z = Encoder(x), shape (L * d)
## 2. the projector converts these into single series level embeddings for each masked version
#     S = Projector(z), shape (d')
## 3. Leads to: S = MLP(s) which is what we are implementing here. 
# --------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F

def projector(input_dim, hidden_dim = None, output_dim = None):
    """
    Builds the shared projecot MLP.

    Args: 
        input_dim: dimension of encoder output, d
        hidden_dim: hidden layer size, default = input_dim
        output_dim: output dimension of projector, default = input_dim
    """
    hidden_dim = hidden_dim or input_dim
    output_dim = output_dim or input_dim

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )


# call it in the training later: 
# projector_output = projector(input_dim = d)
# S = projector_output(seriese_wise)
