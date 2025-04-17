import numpy as np
import torch

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
