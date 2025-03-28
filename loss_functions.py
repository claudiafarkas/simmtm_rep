
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
