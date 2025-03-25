
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

def manifold_constrain_loss(S, tau):
    #will implement later
    pass

def tot_loss(x, x_hat, S, lamb, tau):
    """
    Computes the overall loss.

    Args:
        x: Original time series.
        x_hat: Reconstructed time series.
        S: Series-wise representations.
        lamb: Balance between reconstruction and constraint.
        tau: Temperature for similarity.

    Returns:
        total_loss: Combined loss.
    """
    l_reconstruction = reconstruction_loss(x, x_hat)
    l_constraint = manifold_constraint_loss(S, tau)

    return l_reconstruction + lamb * l_constraint
