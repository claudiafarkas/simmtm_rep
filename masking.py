import numpy as np

def geometric_masking(seq_x, r=0.5, lm=3, seed=None):
    """
    Apply geometric masking to the input sequence.
    Got the formula for geometric masking from 'A TRANSFORMER-BASED FRAMEWORK FOR MULTIVARIATE TIME SERIES REPRESENTATION LEARNING'

    Parameters:
    - seq_x: np.ndarray of shape (seq_len, num_channels)
    - r: proportion of the sequence to be masked (default 0.5)
    - lm: mean length of masked segments (default 3)
    - seed: random seed for reproducibility (optional)

    Returns:
    - masked_seq_x: np.ndarray with masked segments zeroed out
    - mask: np.ndarray indicating masked positions (1 for masked, 0 for unmasked)
    """
    if seed is not None:
        np.random.seed(seed)
    
    seq_len, num_channels = seq_x.shape
    masked_seq_x = seq_x.copy()
    mask = np.zeros_like(seq_x)
  
    lu = int((1 - r) / r * lm)  # mean length of unmasked segments, got the formula from the reference paper 

    for j in range(num_channels):
        pos = 0
        while pos < seq_len:
            # Masked segment
            mask_len = np.random.geometric(1 / lm)
            mask_len = min(mask_len, seq_len - pos)
            masked_seq_x[pos:pos + mask_len, j] = 0
            mask[pos:pos + mask_len, j] = 1
            pos += mask_len

            # Unmasked segment
            unmask_len = np.random.geometric(1 / lu)
            pos += unmask_len

    return masked_seq_x, mask
