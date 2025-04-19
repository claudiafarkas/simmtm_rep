# USAGE IN PAPER DESCRIPTION:
# Described as a process of reconstructing the original time series. 
# The model uses point-wise representation for each masked variant using an encoder, the representations are then aggregated using weights
# that come from the difference between the serise_wise representation and the original masked series.

# Paper's Equation Interpretation Steps (see pg.4 "Point-wise aggregation"):
# 1. Take the seriese-wise similarity R_s_i,s' between the original serise s_i and each masked series s'
# 2. Scale each similarity by doing 1/temperatrue
# 3. Calculate softmax across the scaled similarities to get weights
# 4. Multiply each masked series's point-wise reprsentation z' by the associated softmax wieght and then sum it all up. 
import torch
import torch.nn.functional as F


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
