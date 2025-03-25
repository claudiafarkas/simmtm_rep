
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
