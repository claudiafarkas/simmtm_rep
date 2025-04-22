import os
import sys
HERE = os.path.dirname(__file__)                 # …/simmtm_rep/classification
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from classification_model import SimMTMModelWithResNet, tot_loss, geometric_masking, series_wise_similarity, point_wise_reconstruction
from classification_data_processing import load_pt, normalize_data, dataloader
from common_building_blocks.mlp_decoder import decoder
from common_building_blocks.mlp_projector import projector
from common_building_blocks.loss_functions import tot_loss
from common_building_blocks.point_wise import point_wise_reconstruction
from common_building_blocks.series_wise import series_wise_similarity
from classification.encoder_layer import ResNetEncoder


class CLArgs:
    def __init__(self):
        # data loading
        self.epilepsy_dir = "data/epilepsy/"
        self.emg_dir = "data/emg/"

        # pre‑training (in‑domain)
        self.cls_pretrain_batch_size = 128    
        self.cls_pretrain_lr = 1e-4
        self.cls_pretrain_epochs = 50

        # fine‑tuning (corss‑domain)
        self.cls_finetune_batch_size = 32
        self.cls_finetune_lr = 1e-4
        self.cls_finetune_epochs = 300 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
classArgs = CLArgs()


class SimMTMModelWithResNet(nn.Module):
    def __init__(self, in_channels, resnet_out_dim=256, proj_dim=32, num_masked=3, num_classes = None):
        super(SimMTMModelWithResNet, self).__init__()
        # standard set up - calling all the helper functions
        self.num_masked = num_masked
        self.encoder = ResNetEncoder(in_channels = in_channels)                               # returns (B, resnet_out_dim)
        self.projector = projector(input_dim = resnet_out_dim, output_dim = proj_dim)
        self.decoder = decoder(input_dim = resnet_out_dim, output_dim = in_channels)

        # classification head (only used at fine‐tune time)
        if isinstance(num_classes, int) and num_classes > 0:
            self.cls_head = nn.Linear(resnet_out_dim, num_classes)
        else:
            self.cls_head = None

    # helper function for the pre-training
    def pre_train_forward(self, seq_x):
        """ 
        seq_x: is in the form of (B, L, C) 
        returns: the loss result
        """
        B, L, C = seq_x.shape
        seq_x = seq_x.to(dtype=torch.float32, device=seq_x.device)
        original = seq_x.clone()
       
        # geometric masking call
        masked_views, masks = geometric_masking(original)

        seq_x = seq_x.permute(0, 2, 1) 
        
        masked_views = torch.from_numpy(masked_views).float().to(seq_x.device).permute(0,2,1)           # B*M, C, L

        # stack masked views
        inputs = torch.cat([
            torch.stack([seq_x[i]] + [masked_views[i * self.num_masked + j] for j in range(self.num_masked)], dim=0)
            for i in range(B)
        ], dim=0) 

        # encode
        enc_output = self.encoder(inputs)                                                   
        # print("Encoder output shape:", enc_output.shape) 
        # project
        series_proj = self.projector(enc_output)
        # Similarity matrix
        R = series_wise_similarity(series_proj.mean(dim=1))
        # print("R Size:", R.shape)
        # print("R:", R)
        enc_output = enc_output.unsqueeze(2)    
        aggregated = point_wise_reconstruction(R, enc_output, num_masked = self.num_masked)         # (B, L, D)
        # print("Aggregation Size (pre squeeze): ", aggregated.shape)
        aggregated = aggregated.squeeze(2)
        # print("Aggregated: ", aggregated)
        # print("Aggregation Size (post squeeze): ", aggregated.shape)    

        # decode
        reconstructed = self.decoder(aggregated)
        # print("Reconstruction: ", reconstructed)
        # print("Reconstructed Size: ", reconstructed.shape)

        pooled = aggregated.mean(dim=1)                                                           # ((B*M+1), D)
        # print("Pooled Size: ", pooled.shape)
        
        total_loss = tot_loss(original, reconstructed, R, self.num_masked, lamb=0.1, t=0.1)
        # print("Total loss:", total_loss.item())
        
        # classification of logits
        if self.cls_head is not None:
            logits = self.cls_head(pooled)                                                  
        else:
            logits = None
        return total_loss, reconstructed, R, logits

    # helper function to help compute the finetuning 
    def fine_tune_forward(self, seq_x):
        seq_x = seq_x.to(dtype=torch.float32, device=seq_x.device)                           # (B, C, L)
        # print("SEQ_X Size: ",seq_x.shape)
        seq_x = seq_x.permute(0, 2, 1)
        # Encode
        enc_output = self.encoder(seq_x)                                                     # (B*(M+1), resnet_out_dim)
        # classification of logits
        if enc_output.dim() == 3:
            enc_output = enc_output.mean(dim=1)     
        else: 
            enc_output = enc_output
        
        logits = self.cls_head(enc_output)
        return None, None, None, logits
    
    # calls if finetuning or pretaining is needed                                                    
    def forward(self, seq_x, mode = "pretrain"):
        if mode == "pretrain":
            return self.pre_train_forward(seq_x)
        elif mode == "finetune": 
            return self.fine_tune_forward(seq_x)
        else:
            raise ValueError("Unknown mode: ", mode)                      
    

    def training_step(self, seq_x):
        tot_loss,*_ = self(seq_x)
        return tot_loss
   
                                                     

# ------- EMG Data Loading for Pre-training --------
print("\n")
print("---------------------------|EPI Pre-Training|---------------------------")
# data loading for emg
emg = load_pt(classArgs.emg_dir)        # to get pretraiing for any other data set just replace the load param here 
X_emg_pre = emg["train"]["samples"]
C = X_emg_pre.shape[2]
X_gest_pre = normalize_data(X_emg_pre)#.permute(0,2,1)
pretrain_loader = dataloader(X_gest_pre, torch.zeros(X_gest_pre.size(0)), classArgs.cls_pretrain_batch_size, shuffle=True)


#-------- Pre-Training On EMG--------
emg_pretrain_model = SimMTMModelWithResNet(in_channels = C, num_classes = None).to(classArgs.device)
emg_pretrain_optimizer = optim.Adam(emg_pretrain_model.parameters(), lr = classArgs.cls_pretrain_lr)

#loss calcuation looop
for epoch in range(classArgs.cls_pretrain_epochs):      # will loop for 50 to get the loss per epoch
    emg_pretrain_model.train()
    total_loss = 0.0

    for seq_x, _ in pretrain_loader:
        seq_x = seq_x.float().to(classArgs.device)
        emg_pretrain_optimizer.zero_grad()

        loss = emg_pretrain_model.training_step(seq_x)
        loss.backward()
        emg_pretrain_optimizer.step()
        total_loss += loss.item()
       
    avg_loss = total_loss / len(pretrain_loader)

    print(f"\nEpilepsy Pretrain | Epoch {epoch+1}/ {classArgs.cls_pretrain_epochs} completed — avg loss = {avg_loss:.4f}")

    torch.save(emg_pretrain_model.state_dict(), 'epilepsy_pretrained.pth')

#--------Fine-tune and Evaluate on Epilepsy Data ---------
# finetune data loading
epilepsy = load_pt(classArgs.epilepsy_dir)          # to get pretraiing for any other data set just replace the load param here
# splitting into test and train sets 
X_fine = normalize_data(epilepsy["val"]["samples"])
y_fine = epilepsy["val"]["labels"]
X_test = normalize_data(epilepsy["test"]["samples"])
y_test = epilepsy["test"]["labels"]
num_classes = int(torch.unique(y_fine).numel())
C_fine = X_fine.shape[2]

finetune_loader = dataloader(X_fine, y_fine, classArgs.cls_finetune_batch_size, False)
test_loader = dataloader(X_test, y_test, classArgs.cls_finetune_batch_size, False)

# -------- Fine-Tuning on Epilepsy----------
epi_finetune_model = SimMTMModelWithResNet(in_channels = C_fine , num_classes = num_classes).to(classArgs.device)
pretrained_dict = torch.load('epilepsy_pretrained.pth')
epi_model_dict = epi_finetune_model.state_dict()
compatible_dict = {}

# loop to only compare the weights so so cross domain comaprisson will have the same shape, just makes it easier to use them  
for k, v in pretrained_dict.items():
    if k in epi_model_dict and epi_model_dict[k].shape == v.shape:
        compatible_dict[k] = v
    epi_model_dict.update(compatible_dict)
    epi_finetune_model.load_state_dict(epi_model_dict)

epi_finetune_optimizer = optim.Adam(epi_finetune_model.parameters(), lr=classArgs.cls_finetune_lr)

print("\n")
print("-------------------------|Epilepsy Fine-tuning|------------------------")
print("Epoch | TrnLoss | TrnAcc | ValLoss | ValAcc |  Prec  | Recall |   F1  ")
print("-----------------------------------------------------------------------")

for epoch in range(classArgs.cls_finetune_epochs):
    # seq_x = seq_x.float()
    epi_finetune_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

# loop to calcualte all the metrics of finetunning
# finetuning training loop
    for seq_x, labels in finetune_loader:        
        seq_x, labels = seq_x.to(classArgs.device), labels.to(classArgs.device)
        epi_finetune_optimizer.zero_grad()
        _, _, _, logits = epi_finetune_model(seq_x, mode = "finetune")
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        epi_finetune_optimizer.step()
        
        # accuracy calculations
        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(finetune_loader)
    train_acc = 100 * train_correct / train_total

    # -- evaluation loop -- 
    epi_finetune_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for seq_x, labels in test_loader:
            seq_x, labels = seq_x.float().to(classArgs.device), labels.to(classArgs.device)
            _, _, _, logits = epi_finetune_model(seq_x, mode = "finetune")
            loss = F.cross_entropy(logits, labels)
            
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(test_loader)
    val_acc = accuracy_score(val_labels, val_preds) *100
    val_prec = precision_score(val_labels, val_preds, average='macro', zero_division = 0) *100
    val_recall = recall_score(val_labels, val_preds, average='macro', zero_division = 0) * 100
    val_f1 = f1_score(val_labels, val_preds, average = 'macro', zero_division = 0) * 100

    print(f"{epoch:3d}   | " f"{train_loss:6.4f} | {train_acc:6.2f}%  | " f"{val_loss:6.4f} |  {val_acc:6.2f}% | " f"{val_prec:6.2f}% | {val_recall:6.2f}% | {val_f1:6.2f}%")

print("\nTraining complete! Saving final checkpoint…")

torch.save(epi_finetune_model.state_dict(), 'emg2epilepsy_finetuned.pth')






# -- Copied Over Functions From Other Files in Repo---
# GEOMETRIC MASKING
def geometric_masking(seq_x: torch.Tensor, r=0.5, lm=3, num_views = 3, seed=None):
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

    # batch_size, seq_len, num_channels = seq_x.shape
    B, L, C = seq_x.shape
    lu = max(1, int((1 - r)/ r * lm))

    masked_views = []
    masks = []

    # lu = int((1 - r) / r * lm)  # mean length of unmasked segments, got the formula from the reference paper
    for _ in range(num_views): # for multiple views, multiple layers of masking of the same input
        masked_seq_x = seq_x.clone()
        mask = np.zeros_like(seq_x.detach().numpy())
        for b in range(B):
            for j in range(C): # for channel independence
                pos = 0
                while pos < L:
                    # Masked segment
                    mask_len = np.random.geometric(1 / lm)
                    mask_len = min(mask_len, L - pos)
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




