from simmtm_rep.forecasting.data_loader.data_factory import data_provider
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from simmtm_rep.common_building_blocks.masking import geometric_masking
from simmtm_rep.common_building_blocks.mlp_decoder import decoder
from simmtm_rep.common_building_blocks.mlp_projector import projector
from simmtm_rep.common_building_blocks.loss_functions import tot_loss
from simmtm_rep.common_building_blocks.point_wise import point_wise_reconstruction
from simmtm_rep.common_building_blocks.series_wise import series_wise_similarity
from simmtm_rep.forecasting.encoder_layer import ChannelIndependentTransformer

# for cross_domain training we need to load the weather data
class Args1:
    def __init__(self):
        self.data = 'Weather'  # Using Dataset_weather
        self.root_path = '/home/kristal/Desktop/Pycharm/pythonProject/simmtm_rep/forecasting/dataset/weather/'
        self.data_path = 'weather.csv'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 48
        self.features = 'M'
        self.target = 'OT'
        self.embed = 'timeF'
        self.freq = 't'
        self.task_name = 'forecasting'  # or 'classification' if needed
        self.batch_size = 32
        self.num_workers = 0
        self.seasonal_patterns = None

args1 = Args1()

# Get train data and loader for pre-training for cross-domain
train_dataset, train_loader = data_provider(args1, flag='train')

# loading the ETTm1 data for cross-domain fine-tuning OR in-domain training
class Args2:
    def __init__(self):
        self.data = 'ETTm1'  # Using Dataset_ETT_minute
        self.root_path = '/home/kristal/Desktop/Pycharm/pythonProject/simmtm_rep/forecasting/dataset/ETT-small'
        self.data_path = 'ETTm1.csv'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 48
        self.features = 'M'
        self.target = 'OT'
        self.embed = 'timeF'
        self.freq = 't'
        self.task_name = 'forecasting'  # or 'classification' if needed
        self.batch_size = 32
        self.num_workers = 0
        self.seasonal_patterns = None

args2 = Args2()

# For cross-domain this is our fine-tuning dataset, for in-domain it's both the pretraining and finetuning dataset
f_train_dataset, f_train_loader = data_provider(args2, flag='train') # for pretraining in-domain, 75% of the data
fv_train_dataset, fv_train_loader = data_provider(args2, flag='val') # for fine-tuning, 25% of the data
ft_train_dataset, ft_train_loader = data_provider(args2, flag='test') # for evaluation, 25 % of the data but batch size 1 instead of 32

class SimMTMModel(nn.Module):
    def __init__(self, num_channels, d_model=16, n_head=4, n_layers=2, proj_dim=32, num_masked=3, dropout=0.1):
        super(SimMTMModel, self).__init__()
        self.num_masked = num_masked
        self.encoder = ChannelIndependentTransformer(
            num_channels=num_channels,
            dim=d_model,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout
        )
        self.projector = projector(d_model*num_channels, output_dim=proj_dim)
        self.decoder = decoder(d_model*num_channels, output_dim=num_channels)
        self.d_model = d_model

    def forward(self, seq_x):
        # Geometric masking
        masked_views, masks = geometric_masking(seq_x.cpu())
        masked_views = torch.tensor(masked_views).float().to(seq_x.device)
        inputs = torch.cat([
            torch.stack([seq_x[i]] + [masked_views[i * self.num_masked + j] for j in range(self.num_masked)], dim=0)
            for i in range(seq_x.shape[0])
        ], dim=0)

        # Encoder
        enc_output = self.encoder(inputs)
        # (N*(M+1), L, C, d_model)
        B_total, L, C, d_model = enc_output.shape
        enc_output_f = enc_output.view(B_total, L, -1)  # flatten channels
        # Project to series-wise representations
        series_repr = enc_output_f.mean(dim=1) # (N*(M+1), d_model)
        series_proj = self.projector(series_repr)

        # Similarity matrix
        R = series_wise_similarity(series_proj)

        # Point-wise reconstruction
        reconstructed_z = point_wise_reconstruction(R, enc_output, num_masked=self.num_masked)
        reconstructed_z = reconstructed_z.view(reconstructed_z.shape[0], L, C * d_model)  # (B, L, C*d_model)
        # Decode
        reconstructed = self.decoder(reconstructed_z)
        return seq_x.float(), reconstructed, R

    def training_step(self, seq_x):
        original, reconstructed, R = self(seq_x)
        total_loss = tot_loss(original, reconstructed, R, self.num_masked, lamb=0.1, t=0.02)
        return total_loss

# Forecasting head  ---> the paper doesn't mention anything about this, had to look at one of their reference papers: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
class ForecastingModel(nn.Module):
    def __init__(self, encoder, seq_len, pred_len, d_model, num_channels):
        super().__init__()
        self.encoder = encoder
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len * d_model * num_channels, pred_len * num_channels) # we are basically trying to predict 'the most probable length-O series in the future given the past length-I series'
        self.num_channels = num_channels

    def forward(self, seq_x):
        with torch.no_grad(): # feezing any grad updates
            enc_output = self.encoder(seq_x)  # getting the point-wise rep
        B, L, C, d_model = enc_output.shape
        flat = enc_output.view(B, -1)
        pred = self.linear(flat) # in a way we are projecting the encoder output to the prediction length-O series
        return pred.view(B, self.pred_len, self.num_channels)

# Training pretraining model
task = 'cross_domain' # this is either 'in_domain' or 'cross_domain'
if task == 'in_domain':
    train_loader = f_train_loader
    train_dataset = f_train_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_batch = next(iter(train_loader))[0]
num_channels = sample_batch.shape[-1]
model = SimMTMModel(num_channels=num_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Pre training on the training set of ETTm1 or weather data depending on the setting
for epoch in range(50):
    model.train()
    total_loss = 0
    for seq_x, *_ in train_loader:
        seq_x = seq_x.float().to(device)
        optimizer.zero_grad()
        loss = model.training_step(seq_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Pretraining Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

torch.save(model.encoder.state_dict(), "simmtm_encoder.pth") # update the model name based on the setting 'in_domain' or 'cross_domain'

class ChannelAdapter(nn.Module): # the adapter is for cross-domain training coz the channel numbers don't always match between different datasets
    def __init__(self, in_channels=7, out_channels=21):
        super(ChannelAdapter, self).__init__()
        self.adapter = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, L, C), applies adapter to the last dimension
        return self.adapter(x)

# Forecasting phase 
forecasting_model = ForecastingModel(
    encoder=ChannelIndependentTransformer(num_channels=num_channels, dim=16, n_head=4, n_layers=2).to(device),
    seq_len=96, # given in the dataloader
    pred_len=48, # given in the dataloader
    d_model=16,
    num_channels=num_channels
).to(device)

forecasting_model.encoder.load_state_dict(torch.load("simmtm_encoder.pth"))
forecasting_model.encoder.eval()
if task != 'in_domain': # we need an adapter for weather ---> ETTm1 channel number adaptation
    adapter = ChannelAdapter(in_channels=7, out_channels=21).to(device)
    for param in adapter.parameters():
        param.requires_grad = False
optimizer = optim.Adam(forecasting_model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() # fine tuning is done only using the L2 loss

# Fine tuning on the validation set of ETTm1
for epoch in range(10):
    forecasting_model.train()
    total_loss = 0
    for seq_x, *_ in fv_train_loader: 
        seq_x = seq_x.float().to(device)
        if task != 'in_domain':
            seq_x = adapter(seq_x)
        optimizer.zero_grad()
        pred = forecasting_model(seq_x)
        loss = loss_fn(pred, seq_x[:,-49:-1,:]) # we are getting the last 48 segments from the sequence to compare it to the prediction
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Forecasting Epoch {epoch+1}: Loss = {total_loss / len(fv_train_loader):.4f}")

# Evaluation on the test set of ETTm1
forecasting_model.eval()
mae_total, mse_total = 0, 0
with torch.no_grad():
    for seq_x, *_  in ft_train_loader:
        seq_x = seq_x.float().to(device)
        if task != 'in_domain':
            seq_x = adapter(seq_x)
        pred = forecasting_model(seq_x)
        mse = F.mse_loss(pred, seq_x[:,-49:-1,:]).item()
        mae = F.l1_loss(pred, seq_x[:,-49:-1,:]).item()
        mse_total += mse
        mae_total += mae
print(f"Final Test MSE: {mse_total / len(ft_train_loader):.4f}, MAE: {mae_total / len(ft_train_loader):.4f}")

