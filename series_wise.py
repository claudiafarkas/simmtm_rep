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
