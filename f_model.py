class SimMTMModel_Forecasting(nn.Module):
    def __init__(self, num_channels, d_model=64, n_head=4, n_layers=4, proj_dim=32, num_masked=3, dropout=0.1):
        super(SimMTMModel_Forecasting, self).__init__()
        self.num_masked = num_masked
        self.encoder = ChannelIndependentTransformer(
            num_channels=num_channels,
            dim=d_model,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout
        )
        self.projector = projector(d_model*num_channels, output_dim=proj_dim)
        self.decoder = decoder(d_model, output_dim=num_channels)
        self.d_model = d_model

    def forward(self, seq_x):
        # Geometric masking
        masked_views, masks = geometric_masking(seq_x.cpu()) # shape: B*num_masked,L,C for the masking
        masked_views = torch.tensor(masked_views).float().to(seq_x.device)
        inputs = torch.cat([
            torch.stack([seq_x[i]] + [masked_views[i * self.num_masked + j] for j in range(self.num_masked)], dim=0)
            for i in range(seq_x.shape[0])
        ], dim=0)  # [x0, x0_mask1, x0_mask2, ..., xN-1, xN-1_mask1, xN-1_mask2, ...] shape: B*(1+num_masked),L,C 

        # Encoder
        enc_output = self.encoder(inputs)  # shape: B*(1+num_masked),L,C,d_model
        B_total, L, C, d_model = enc_output.shape
        enc_output = enc_output.view(B_total, L, -1)  # flatten channels - shape: B*(1+num_masked),L,C*d_model

        # Project to series-wise representations
        series_repr = enc_output.mean(dim=1)  # (N*(M+1), d_model) shape: B*(1+num_masked), C*d_model --> doing this coz paper says the input series_wise_similarity is 1 x d
        series_proj = self.projector(series_repr) # shape: B*(1+num_masked), proj_dim

        # Similarity matrix
        R = series_wise_similarity(series_proj) # shape B*(1+num_masked), B*(1+num_masked) --> we get this shape after S*S.T, remember the order of S is [x0, x0_mask1, x0_mask2, ..., xN-1, xN-1_mask1, xN-1_mask2, ...]

        # Point-wise reconstruction
        z_point = enc_output.view(B_total, L, C, d_model)
        z_point = z_point.view(-1, self.num_masked + 1, L, C,d_model)  # reshape into (B, M+1, L,C, d_model)
        aggregated = point_wise_reconstruction(R, z_point, tau=0.1) # this isn't working rn

        # Decode
        reconstructed = self.decoder(aggregated)

        # Compute loss
        original = seq_x
        total_loss = tot_loss(original, reconstructed, R, self.num_masked, lamb=0.1, t=0.1)

        return total_loss, reconstructed
