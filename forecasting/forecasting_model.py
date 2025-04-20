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
train_dataset, train_loader = data_provider(args1, flag='train') # for pretraining cross-domain, 78% of the data
t_train_dataset, t_train_loader = data_provider(args1, flag='test') # for evaluating the cross-domain pretraining, 22% of the data
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

    def evaluate_step(self, seq_x):
        self.eval()
        with torch.no_grad():
            original, reconstructed, _ = self(seq_x)
            mse = F.mse_loss(reconstructed, original, reduction='mean').item()
            mae = F.l1_loss(reconstructed, original, reduction='mean').item()
        return mse, mae

task = 'in_domain' # this is either 'in_domain' or 'cross_domain'
if task == 'in_domain':
    train_loader = f_train_loader
    train_dataset = f_train_dataset
    t_train_loader = ft_train_loader
    t_train_dataset = ft_train_dataset

sample_batch = next(iter(train_loader)) # --> the weather data for the cross-domain task
seq_x = sample_batch[0]  # assuming (seq_x, _, _) format
num_channels = seq_x.shape[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimMTMModel(num_channels=num_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#pre-training
for epoch in range(50):
    model.train()
    total_loss = 0.0
    counter_t = 0
    for seq_x, *_ in train_loader:
        seq_x = seq_x.float().to(device)
        optimizer.zero_grad()
        loss = model.training_step(seq_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        counter_t += 1

    # Evaluation
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    counter = 0
    for seq_x, *_ in t_train_loader:
        seq_x = seq_x.float().to(device)
        mse, mae = model.evaluate_step(seq_x)
        total_mse += mse
        total_mae += mae
        counter += 1

    print(f"Pretrain Epoch [{epoch + 1}/50] | Loss: {total_loss/counter_t:.4f} | Avg MSE: {total_mse / counter:.4f} | Avg MAE: {total_mae / counter:.4f}")
torch.save(model.state_dict(), 'pretrained_simmtm_model.pth')


class ChannelAdapter(nn.Module): # the adapter is for cross-domain training coz the channel numbers don't always match between different datasets
    def __init__(self, in_channels=7, out_channels=21):
        super(ChannelAdapter, self).__init__()
        self.adapter = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (B, L, C), apply adapter to the last dimension
        return self.adapter(x)


# fine-tune
finetune_model = SimMTMModel(num_channels=num_channels).to(device)
finetune_model.load_state_dict(torch.load('pretrained_simmtm_model.pth'))  # Start from pretrained weights, change the name of the model depending on what u want to load
if task != 'in_domain': # we need an adapter for weather ---> ETTm1 channel number adaptation
    adapter = ChannelAdapter(in_channels=7, out_channels=21).to(device)
optimizer = optim.Adam(finetune_model.parameters(), lr=1e-4)

for epoch in range(10):
    finetune_model.train()
    total_loss = 0.0
    counter_t = 0
    for seq_x, *_ in fv_train_loader:
        seq_x = seq_x.float().to(device)
        if task != 'in_domain':
            seq_x = adapter(seq_x)
        optimizer.zero_grad()
        original, reconstructed, _ = finetune_model(seq_x)
        loss = F.mse_loss(reconstructed, original)  # finetune with L2 only
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        counter_t +=1
    # Evaluation
    finetune_model.eval()
    total_mse, total_mae = 0.0, 0.0
    counter = 0
    for seq_x, *_ in ft_train_loader:
        seq_x = seq_x.float().to(device)
        if task != 'in_domain':
            seq_x = adapter(seq_x)
        mse, mae = finetune_model.evaluate_step(seq_x)
        total_mse += mse
        total_mae += mae
        counter += 1

    print(
        f"Finetune Epoch [{epoch + 1}/10] | Loss: {total_loss / counter_t:.4f} | Avg MSE: {total_mse / counter:.4f} | Avg MAE: {total_mae / counter:.4f}")

