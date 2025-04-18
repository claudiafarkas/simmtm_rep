import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# classification = no masking, just one series-level vector per input
class SimMTMModel_Classification(nn.Module):
    def __init__(self, num_channels, d_model=16, num_classes=5, proj_dim=32, num_masked=3):
        super(SimMTMModel_Classification, self).__init__()
        self.num_masked = num_masked
        self.d_model = d_model
        self.num_channels = num_channels

        # Encoder outputs
        self.encoder = ResNetEncoder(
            in_channels = num_channels,
            base_channels = d_model,
            block_counts = [2, 2, 2]
        )

        self.projector = projector(input_dim=7, output_dim=proj_dim)
        self.decoder = build_decoder(d_model * num_channels, output_dim=num_classes)
        self.d_model = d_model


    def forward(self, seq_x, labels = None):
        # Geometric masking
        masked_views, masks = geometric_masking(seq_x.cpu())
        masked_views = torch.tensor(masked_views).float().to(seq_x.device)
        # shape: (B * (M+1), L, C)
        inputs = torch.cat([
            torch.stack([seq_x[i]] + [masked_views[i * self.num_masked + j] for j in range(self.num_masked)], dim=0)
            for i in range(seq_x.shape[0])
        ], dim=0)

        B_total = inputs.shape[0]
        # B = seq_x.shape[0]
        M_plus_1 = self.num_masked + 1
        B = B_total // M_plus_1

        # Encoder
        enc_output = self.encoder(inputs.permute(0, 2, 1))      # ResNet expects (B, C, L)
        enc_output = enc_output.permute(0, 2, 1)                # back to (B_total, L, d_model)
        print("Encoder output shape:", enc_output.shape)
        print("Encoder output sample:", enc_output[0]) 
        # L, D = enc_output.shape[1], enc_output.shape[2]
        
        # project to series-wise representation
        series_repr = enc_output.mean(dim=1)                     # (B_total, d_model)
        print("series_repr shape:", series_repr.shape)           # Should be (B_total, D_actual)
        series_proj = self.projector(series_repr)                # (B_total, proj_dim)
        series_proj = series_proj.view(B, M_plus_1, -1)          # reshape: (B, M+1, proj_dim)
        
        # Similarity matrix (per batch)
        R = torch.stack([                                      
            series_wise_similarity(series_proj[i]) for i in range(B)
        ], dim=0)  # (B, M+1, M+1)
        print("R Size:", R.shape)
        print("R:", R)

        # point-wise reconstruction 
        enc_output = enc_output.view(B, M_plus_1, enc_output.shape[1], enc_output.shape[2])  # (B, M+1, L, D)
        aggregated = point_wise_reconstruction(R, enc_output, tau=0.1, num_masked = self.num_masked)  # (B, L, D)

        # classification of logits
        pooled = aggregated.mean(dim=1)  # (B, D)
        logits = self.decoder(pooled)    # (B, num_classes)


        # Decode
        reconstructed = self.decoder(aggregated)

        i = 0  # pick an example from the batch
        # original_example = seq_x[i].detach().cpu().numpy()
        # recon_example = reconstructed[i].detach().cpu().numpy()

          # Compute loss
        original = seq_x
        total_loss = tot_loss(original, reconstructed, R, self.num_masked, lamb=0.1, t=0.1)
        print("Total loss:", total_loss.item())

        return total_loss, reconstructed

        # # Reshape Point-wise reconstruction
        # reconstructed_z = point_wise_reconstruction(R, enc_output, num_masked=self.num_masked)
        # # reconstructed_z = reconstructed_z.view(reconstructed_z.shape[0], L, C * d_model)  # (B, L, C*d_model)
        # # Decode
        # reconstructed = self.decoder(reconstructed_z)
        # return seq_x.float(), reconstructed, R
    

    

# -- copied functions ---

#DECODER - SHARED
def build_decoder(input_dim, output_dim, hidden_dim = None):
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


def apply_decoder(z_hat, decoder):
    """
    Applies the decoder.
    
    Args:
        z_hat: the aggregated point-wise time step 
        decoder (nn.Module): the decoder MLP
    """
    return decoder(z_hat)


# ENCODER - C
class Conv1D(torch.nn.Module):
    """
    Main use case is used to extract features (point-wise temporal) from the time-series data like EEG.
    1D Convolutional block that implements: Conv1d, Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Conv1D, self).__init__()
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias = False)       # applies 1D conv on the input
        self.bn = torch.nn.BatchNorm1d(out_channels)                                                             # normalizes for training
        self.relu = torch.nn.ReLU(inplace = True)                                                                # applies relu activation
        

    def forward(self, x):                                                                                        # put its together to put conv, batch norm then relu
        out = self.conv(x)
        out = self.bn(out)
        return self.relu(out)


class Residual1D(torch.nn.Module):
    """
    Implements the 'residual learning' from ResNEt, it helps the network to not overfit and get stuck while training.
    Contains:
        2 conv layers and a skip connection whihc helps stablalize the deep networks.
    """

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Residual1D, self).__init__()
        
        self.conv1 = Conv1D(in_channels, out_channels, kernel_size, stride, padding)                            # 1st conv block; no downsampling
        self.conv2 = Conv1D(out_channels, out_channels, kernel_size, stride = 1, padding = padding)              # 2nd conv block; maintinas the dimension
        self.downsample = None
    
        if in_channels != out_channels or stride != 1:                                                          # identifies if input and output dimensions match, if not then adjust 
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False), 
                torch.nn.BatchNorm1d(out_channels))
        

    def forward(self, x):
        out = self.conv1(x)
        print("Shape before conv2:", out.shape)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return F.relu(out)
    

class ResNetEncoder(torch.nn.Module):
    """
    The encoder extracts features by using an initial convolution and pooling method.
    Then, it stacks the residual blocks to learn. As the last step, it applies global avg pooling to 
    create a fixed-length feature vector.   
    """
    def __init__(self, in_channels = 1, base_channels = 64, block_counts = [2, 2, 2]):
        super(ResNetEncoder, self).__init__()
        
        # initial convolution and max pooling 
        self.inital_conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, base_channels, kernel_size = 7, stride= 2, padding = 3, bias = False), 
            torch.nn.BatchNorm1d(base_channels),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool1d(kernel_size = 3, stride = 2, padding =1)
            )
        self.layer1 = self._make_layer(base_channels, base_channels, block_counts[0], stride = 1)                       # step 1: residual blocks - might start with downsampling
        self.layer2 = self._make_layer(base_channels, base_channels * 2, block_counts[1], stride = 2)                   # step 2: residual blocks - downsampling to 2x the channels
        self.layer3 = self._make_layer(base_channels * 2 , base_channels * 4 , block_counts[2], stride = 2)             # step 3: residual blocks - downsampling to 4x the channels
        self.global_avg_pooling = torch.nn.AdaptiveAvgPool1d(1)                                                         # helps compress the time dimension to be 1 vector / series

    # helper function to build residual blocks
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Residual1D(in_channels, out_channels, stride = stride))                              # 1st block: might start with downsampling
        for _ in range(1, blocks):                                                                         # rest of the block keep the length and # of channels
            layers.append(Residual1D(out_channels, out_channels))
            return torch.nn.Sequential(* layers)
    

    def forward(self, x):        # B, C, L
        # x = x.permute(0, 2, 1)      # B, L C -> B, C L
        x = self.inital_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1)      # back to B, L, C
        # pooled = self.global_avg_pooling(x)
        # flattened = pooled.view(pooled.size(0), -1)
        return x

    
# PROJECTOR
def projector(input_dim, hidden_dim = None, output_dim = None):
    """
    Builds the shared projector MLP.

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


# GEOMETRIC MASKING
def geometric_masking(seq_x, r=0.5, lm=3, num_views = 3, seed=None):
    """
    Apply geometric masking to the input sequence.
    Got the formula for geometric masking from 'A TRANSFORMER-BASED FRAMEWORK FOR MULTIVARIATE TIME SERIES REPRESENTATION LEARNING'

    Parameters:
    - seq_x: np.ndarray of input (batch_size, seq_len, num_channels)
    - r: proportion of the sequence to be masked (default 0.5)
    - lm: mean length of masked segments (default 3)
    - num_views: how many times the input is going to be masked, gives us the 'multiple views' aspect the paper talks about
    - seed: random seed for reproducibility (optional)

    Returns:
    - masked_seq_x: np.ndarray with masked segments zeroed out
    - mask: np.ndarray indicating masked positions (1 for masked, 0 for unmasked)
    """
    if seed is not None:
        np.random.seed(seed)

    batch_size, seq_len, num_channels = seq_x.shape
    masked_views = []
    masks = []

    lu = int((1 - r) / r * lm)  # mean length of unmasked segments, got the formula from the reference paper
    for view in range(num_views): # for multiple views, multiple layers of masking of the same input
        masked_seq_x = seq_x.clone()
        mask = np.zeros_like(seq_x.detach().numpy())
        for b in range(batch_size):
            for j in range(num_channels): # for channel independence
                pos = 0
                while pos < seq_len:
                    # Masked segment
                    mask_len = np.random.geometric(1 / lm)
                    mask_len = min(mask_len, seq_len - pos)
                    masked_seq_x[b, pos:pos + mask_len, j] = 0
                    mask[b, pos:pos + mask_len, j] = 1
                    pos += mask_len

                    # Unmasked segment
                    unmask_len = np.random.geometric(1 / lu)
                    pos += unmask_len
        masked_views.append(masked_seq_x.detach().numpy())
        masks.append(mask)
    masked_views = np.concatenate(masked_views, axis=0)
    masks = np.concatenate(masks, axis=0)

    return masked_views, masks


# POINT WISE
def point_wise_reconstruction(R, z_input,num_masked, tau=0.1):
    """
    Reconstruct point-wise features for each original series (no self in aggregation).

    Args:
        R (Tensor): similarity matrix
        z_point (Tensor): point-wise representations, encoder_output
        tau (float): softmax temperature

    Returns:
        aggregated (Tensor): (B, L, C, D), aggregated point-wise representations
    """
    # B_total, L, C, D = z_input.shape
    # M_plus_1 = num_masked + 1
    # B = B_total // M_plus_1

    # z_flat = z_input  # (B_total, L, C, D)
    # aggregated = []
    # print("B_total:", B_total, "M+1:", M_plus_1, "→ B:", B)
    
    # for i in range(B): # we are getting the z_i, so iterating for the og time series
    #     anchor_idx = i * M_plus_1  # original series index
    #     masked_idx = [anchor_idx + j + 1 for j in range(num_masked)]

    #     masked_sims = R[i, 0, 1:]
    #     weights = F.softmax(masked_sims / tau, dim = 0)

    #     masked_z = z_flat[masked_idx]
    #     agg = torch.sum(weights.view(-1, 1, 1, 1) * masked_z, dim = 0)


    #     aggregated.append(agg)

    # return torch.stack(aggregated, dim=0)  # (B, L, C, D)

    B_total, L, C, D = z_input.shape
    M_plus_1 = num_masked + 1
    B = B_total // M_plus_1

    z_flat = z_input  # (B_total, L, C, D)
    aggregated = []

    for i in range(B): # we are getting the z_i, so iterating for the og time series
        anchor_idx = i * M_plus_1  # original series index
        sim_row = R[anchor_idx].clone()

        # exclude self, we don't want it in the softmax calculations
        sim_row[anchor_idx] = -float('inf')
        weights = F.softmax(sim_row / tau, dim=0)  # (B_total,)

        # weighted sum of point-wise features, so we just multiply each pointwise with its similarity softmax
        agg = torch.sum(weights.view(-1, 1, 1, 1) * z_flat, dim=0)  # (L, C, D)
        aggregated.append(agg)

    return torch.stack(aggregated, dim=0)  # (B, L, C, D)

#SERIES WISE
def series_wise_similarity(S):
    """
    Calculate pairwise cosine similarity matrix for series-wise representations.

    Args:
    - S: Series-wise representations.

    Returns:
    - R: Pairwise similarity matrix.
    """
    # Dot product between all pairs of series-wise representations
    dot_p = torch.matmul(S, S.T) # so it's technically the u * v^T part

    # Norm of each series-wise representation
    norms = torch.norm(S, dim=1, keepdim=True)  # so basically ||u|| and/or ||v||

    # Cosine similarity calculation
    R = dot_p / (torch.matmul(norms, norms.T) + 1e-8)  # calculating the cosine similarity matrix, 1e-8 is there just in case

    return R 


# LOSS
def reconstruction_loss(x, x_hat):
    """
    Computes the reconstruction loss L_reconstruction.

    Args:
        x: Original time series.
        x_hat: Reconstructed time series.

    Returns:
        loss: Mean squared error loss.
    """
    return torch.mean((x - x_hat) ** 2)


def manifold_constraint_loss(R, M, t=0.1):
    """
    Computes the constraint loss using the similarity matrix.

    Args:
        R: Pairwise similarity matrix we get from series-wise similarity,(N(M+1) * N(M+1))
        M: Number of masked versions of the time series, so num_masked
        t: Temperature scaling factor for contrastive loss

    Returns:
        constraint_l: The constraint loss.
    """
    D = R.shape[0]
    N = D // (M + 1)  # the number of original time series

    loss = 0.0

    # Loop over each x_i
    for i in range(N):
        start_idx = i * (M + 1)  # the index of the current og_time series in R
        end_idx = start_idx + M + 1
        pos_indices = list(range(start_idx+1, end_idx)) # the indices of its own masked versions, the positive similarities

        positive_similarities = torch.exp(  # getting all the s+ (s') for x_i
            R[start_idx, pos_indices] / t
        )

        denom_indices = [j for j in range(D) if j != start_idx] # s'' for x_i
        denom_val = torch.exp(
            R[start_idx, denom_indices] / t
        )
        denom = denom_val.sum()
        series_loss = 0
        for p in positive_similarities:
            series_loss += torch.log(p/(denom+1e-8))
        loss += series_loss
    constraint_l = -(loss / N)  # averaging per sample size, I know the paper doesn't tell us to do this but imma do it anyway
    return constraint_l



def tot_loss(x, x_hat, R, M, lamb, t):
    """
    Computes the overall loss.

    Args:
        x: Original time series
        x_hat: Reconstructed time series
        R: Pairwise similarity matrix
        M: Number of masked versions of the time series, so num_masked
        lamb: Balance between reconstruction and constraint
        t: Temperature for similarity

    Returns:
        total_loss: Combined loss
    """
    l_reconstruction = reconstruction_loss(x, x_hat)
    l_constraint = manifold_constraint_loss(R, M, t)

    return l_reconstruction + lamb * l_constraint




if __name__ == "__main__":
    import torch
    from torch import nn
    import matplotlib.pyplot as plt

    # Dummy inputs
    B, L ,C = 32, 100, 1  # batch size, length, channels
    dummy_input = torch.randn(B, L, C)
    # dummy_input.shape[0] >= 2

    model = SimMTMModel_Classification(num_channels = C)
    model.eval()  # or model.train() for gradient tracking

    with torch.no_grad():  # skip if you want gradients
        loss, recon = model(dummy_input)

    print("Loss:", loss.item())
    # Plot original vs reconstructed for sanity check
    # i = 0
    # plt.plot(dummy_input[i].squeeze().numpy(), label="Original")
    # plt.plot(recon[i].squeeze().numpy(), label="Reconstructed")
    # plt.legend()
    # plt.title("Debug: Original vs Reconstructed")
    # plt.show()