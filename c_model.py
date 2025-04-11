import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SimMTMModel_Classification(nn.Module):
    def __init__(self, in_channels, d_model=64, proj_dim=32, num_masked=3):
        super(SimMTMModel_Classification, self).__init__()
        self.num_masked = num_masked
        self.encoder = ResNetEncoder(
            in_channels = in_channels,
            base_channels = d_model,
            block_counts = [2, 2, 2]
            # dim=d_model,
            # n_head=n_head,
            # n_layers=n_layers,
            # dropout=dropout
        )
        # self.projector = projector(d_model, output_dim=proj_dim)
        self.decoder = decoder(d_model, output_dim=in_channels)
        self.d_model = d_model          # projector output dimension

    def forward(self, seq_x):
        # Geometric masking
        masked_views, masks = geometric_masking(seq_x.cpu())
        masked_views = torch.tensor(masked_views).float().to(seq_x.device)
        inputs = torch.cat([seq_x] + [masked_views[i::self.num_masked] for i in range(self.num_masked)], dim=0)

        # Encoder
        enc_output = self.encoder(inputs)  # (N*(M+1), L, C, d_model)
        B_total, L, feature_dim = enc_output.shape          #rmved C

        # Project to series-wise representations
        series_repr = enc_output.mean(dim=1)  # (N*(M+1), d_model)
        if series_repr.dim() == 1:
            series_repr = series_repr.unsqueeze(0)
        print("series representation shape: ", series_repr.shape)


        # Build projector dynamically
        projector_layer = projector(feature_dim, output_dim=self.d_model)
        series_proj = projector_layer(series_repr)

        B = B_total // (self.num_masked + 1)
        series_proj_reshaped = series_proj.view(B, self.num_masked +1, -1)
        # Similarity matrix
        # R = series_wise_similarity(series_proj)
        R = torch.stack([series_wise_similarity(series_proj_reshaped[i]) for i in range(B)], dim = 0)
        print("R:", R)

        # Point-wise reconstruction
        # reshape the encoder output to goup the viewa per original series.
        # assuption: encoder output has shape (B*(M+1), L, feature_dim)
        # z_point = enc_output.view(B_total, L, feature_dim)  #rmved C
        # z_point = z_point.view(-1, self.num_masked + 1, L, feature_dim)  # reshape into (B, M+1, L,C, d_model)   #rmved C
        z_point = enc_output.view(B, self.num_masked + 1, L, feature_dim)
        pointwise_proj = nn.Linear(feature_dim, self.d_model).to(z_point.device)
        z_point_projected = pointwise_proj(z_point)
        # aggregated = point_wise_reconstruction(series_proj_reshaped, z_point, tau=0.1) # this isn't working rn
        aggregated = point_wise_reconstruction(series_proj_reshaped, z_point_projected, tau=0.1) # working now!

        # Decode
        reconstructed = self.decoder(aggregated)

      

        i = 0  # pick an example from the batch
        original_example = seq_x[i].detach().cpu().numpy()
        recon_example = reconstructed[i].detach().cpu().numpy()

        plt.plot(original_example[0], label='Original')
        plt.plot(recon_example[0], label='Reconstructed')
        plt.legend()
        plt.title("Original vs Reconstructed")
        plt.show()

        # Compute loss
        original = seq_x
        total_loss = tot_loss(original, reconstructed, R, self.num_masked, lamb=0.1, t=0.1)
        print("Total loss:", total_loss.item())

        return total_loss, reconstructed







# ---copied over functions----


def point_wise_reconstruction(series_wise_batch, point_wise_batch, tau=0.1):
    """
    Reconstructs the point-wise representation of the original time serise by aggregating 
    point-wise representations from multiple masked series using weights from serise_wise.py
    B = batch size, M+1 = total # of serise, d = dimension of each serise

    Args:
        serise_wise: Tensor of shape (B, M+1, d)
        point_wise: Tensor of shape (B, M+1, L, d) 
        tau: Temperatrue parameter
    """
    B, M_plus, L, d = point_wise_batch.shape
    reconstructed_list = []

    for i in range(B):
        S = series_wise_batch[i]
        R = series_wise_similarity(S)                       # getting the similarities with M+1 masked versions
        M = M_plus -1                                       # number of masked series (original serise + M masked)

        similarity = R[0,1:]                                # basically just R[i(index for s_i), j(1..M masked s_j)] 
        print("Similarity:", similarity)
        similarity = similarity / tau                       # temperature and scaling (numerator part of the equation)
        
        weights = F.softmax(similarity, dim = 0)            # sum of the exp values (denominator part of the equation)
        point_masked = point_wise_batch[i, 1:, :, :]        # point-wise representation for each masked serise 1...M
        weights_expanded = weights.view(M, 1, 1)            # summing M dimension(the sum part of the equation)
        
        aggregated = torch.sum(weights_expanded * point_masked, dim = 0)        # finds the weighted sum over the M masked serise
        reconstructed_list.append(aggregated)
    
    reconstructed = torch.stack(reconstructed_list, dim = 0)        # final reconstructed shape being: (B, L, d)
    print("Similarity weights:", weights)
    print("Sum of weights:", torch.sum(weights))  # should be ~1
    return reconstructed


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
        mask = np.zeros_like(seq_x)
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
        masked_views.append(masked_seq_x)
        masks.append(mask)
    masked_views = np.concatenate(masked_views, axis=0)
    masks = np.concatenate(masks, axis=0)

    return masked_views, masks



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
    N = D // (M + 1)  # also the batch size technically

    loss = 0.0
    
    # Loop over each x_i
    for i in range(N):

        # I NEED TO DOUBLE-CHECK IF THESE ARE CURRENT BY DEBUGGING
        start_idx = i * (M + 1)  # where the current series masked version start
        end_idx = start_idx + M + 1
        pos_indices = list(range(start_idx, end_idx))

        positive_similarities = torch.exp(  # getting all the s+ (s') for x_i
            R[start_idx, pos_indices] / t
        )  # DOUBLE-CHECK IF THESE ARE THE CORRECT ONES

        negative_indices = [j for j in range(D) if j not in pos_indices]  # s'' for x_i
        negative_similarities = torch.exp(
            R[start_idx, negative_indices] / t
        ) # DOUBLE-CHECK IF start_idx is how we should start
        # remember ---> s′′∈S\{si}

        numr = positive_similarities.sum()
        denom = negative_similarities.sum()
        # contrastive loss for x_i
        series_loss = torch.log(numr / (denom + 1e-8))  # the 1e-8 is to prevent zero-div
        loss += series_loss  
    constraint_l = -(loss/N)  # averaging per sample, and adding that -
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


# call it in the training later: 
# projector_output = projector(input_dim = d)
# S = projector_output(seriese_wise)



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


def apply_decoder(z_hat, decoder):
    """
    Applies the decoder.
    
    Args:
        z_hat: the aggregated point-wise time step 
        decoder (nn.Module): the decoder MLP
    """
    return decoder(z_hat)



# class ConvolutionalPositionalEncoding(nn.Module): # they don't mention anything about positional encoding in the paper but I'm adding one that's similar to Wav2ev2's
#     def __init__(self, dim, kernel, dropout):
#         super(ConvolutionalPositionalEncoding, self).__init__()
#         self.conv = nn.Conv1d(
#             in_channels=dim,
#             out_channels=dim,
#             kernel_size=kernel,
#             padding='same',
#             groups=dim  # depthwise convolution
#         )
#         self.dropout = nn.Dropout(dropout)
#         nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
#         nn.init.constant_(self.conv.bias, 0)
#         self.activation = nn.GELU() 

#     def forward(self, x):
#         x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
#         x = x + self.activation(x_conv)
#         return self.dropout(x)


# class VanillaTransformerEncoder(nn.Module):
#     def __init__(self, num_channels, dim, n_head, n_layers, dim_ff=512, dropout=0.1,
#                  kernel_size=3):
#         super(VanillaTransformerEncoder, self).__init__()
#         self.input_linear = nn.Linear(num_channels, dim)
#         self.conv_pos_encoder = ConvolutionalPositionalEncoding(dim, kernel_size=kernel_size, dropout=dropout)

#         encoder_layer = nn.TransformerEncoderLayer( # a simple vanill;a transformer
#             d_model=dim,
#             nhead=n_head,
#             dim_feedforward=dim_ff,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.d_model = dim

#     def forward(self, x):
#         x = self.input_linear(x) 
#         x = self.conv_pos_encoder(x)  # add conv-based positional encoding
#         x = self.transformer_encoder(x)
#         return x




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
    

#-- fixed to return point-wise representation istead of global
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
        # self.global_avg_pooling = torch.nn.AdaptiveAvgPool1d(1)                                                         # helps compress the time dimension to be 1 vector / series

    # helper function to build residual blocks
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(Residual1D(in_channels, out_channels, stride = stride))                              # 1st block: might start with downsampling
        for _ in range(1, blocks):                                                                         # rest of the block keep the length and # of channels
            layers.append(Residual1D(out_channels, out_channels))
            return torch.nn.Sequential(* layers)
    

    def forward(self, x):
        x = self.inital_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0,2,1)
        # pooled = self.global_avg_pooling(x)
        # flattened = pooled.view(pooled.size(0), -1)
        return x

    




if __name__ == "__main__":
    import torch
    from torch import nn
    import matplotlib.pyplot as plt

    # Dummy inputs
    B, C, L = 2, 100, 1  # batch size, length, channels
    dummy_input = torch.randn(B, C, L)

    model = SimMTMModel_Classification(in_channels = C)
    model.eval()  # or model.train() for gradient tracking

    with torch.no_grad():  # skip if you want gradients
        loss, recon = model(dummy_input)

    print("Loss:", loss.item())
    # Plot original vs reconstructed for sanity check
    i = 0
    plt.plot(dummy_input[i].squeeze().numpy(), label="Original")
    plt.plot(recon[i].squeeze().numpy(), label="Reconstructed")
    plt.legend()
    plt.title("Debug: Original vs Reconstructed")
    plt.show()