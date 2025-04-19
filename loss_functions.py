import torch


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
