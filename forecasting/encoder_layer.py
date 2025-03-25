class ConvolutionalPositionalEncoding(nn.Module): # they don't mention anything about positional encoding in the paper but I'm adding one that's similar to Wav2ev2's
    def __init__(self, dim, kernel, dropout):
        super(ConvolutionalPositionalEncoding, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel,
            padding='same',
            groups=dim  # depthwise convolution
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.conv.bias, 0)
        self.activation = nn.GELU() 

    def forward(self, x):
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.activation(x_conv)
        return self.dropout(x)


class VanillaTransformerEncoder(nn.Module):
    def __init__(self, num_channels, dim, n_head, n_layers, dim_ff=512, dropout=0.1,
                 kernel_size=3):
        super(VanillaTransformerEncoder, self).__init__()
        self.input_linear = nn.Linear(num_channels, dim)
        self.conv_pos_encoder = ConvolutionalPositionalEncoding(dim, kernel_size=kernel_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer( # a simple vanill;a transformer
            d_model=dim,
            nhead=n_head,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = dim

    def forward(self, x):
        x = self.input_linear(x) 
        x = self.conv_pos_encoder(x)  # add conv-based positional encoding
        x = self.transformer_encoder(x)
        return x
