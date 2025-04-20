import os
import sys
HERE = os.path.dirname(__file__)                 # …/simmtm_rep/classification
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from classification_model import SimMTMModelWithResNet, tot_loss, geometric_masking, series_wise_similarity, point_wise_reconstruction
from classification_data_processing import load_pt, normalize_data, dataloader
# from simmtm_rep.common_building_blocks.masking import geometric_masking
from common_building_blocks.mlp_decoder import decoder
from common_building_blocks.mlp_projector import projector
from common_building_blocks.loss_functions import tot_loss
from common_building_blocks.point_wise import point_wise_reconstruction
from common_building_blocks.series_wise import series_wise_similarity
from classification.encoder_layer import ResNetEncoder


class CLArgs:
    def __init__(self):
        self.epilepsy_dir = "data/epilepsy/"
        self.sleepeeg_dir = "data/sleepEEG/"

        # pre‑training (in‑domain)
        self.cls_pretrain_batch_size = 128    
        self.cls_pretrain_lr = 1e-4
        self.cls_pretrain_epochs = 50

        # fine‑tuning (in‑domain)
        self.cls_finetune_batch_size = 32
        self.cls_finetune_lr = 1e-4
        self.cls_finetune_epochs = 300 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
classArgs = CLArgs()


class SimMTMModelWithResNet(nn.Module):
    def __init__(self, in_channels, resnet_out_dim=256, proj_dim=32, num_masked=3, num_classes = None):
        super(SimMTMModelWithResNet, self).__init__()
        self.num_masked = num_masked
        self.encoder = ResNetEncoder(in_channels = in_channels)                               # returns (B, resnet_out_dim)
        self.projector = projector(input_dim = resnet_out_dim, output_dim = proj_dim)
        self.decoder = decoder(input_dim = resnet_out_dim, output_dim = in_channels)

        # classification head (only used at fine‐tune time)
        if isinstance(num_classes, int) and num_classes > 0:
            self.cls_head = nn.Linear(resnet_out_dim, num_classes)
        else:
            self.cls_head = None


    def forward(self, seq_x):
        seq_x = seq_x.to(dtype=torch.float32, device=seq_x.device)
        original = seq_x.clone()
        B, L, C = seq_x.shape
        
        # Geometric masking
        masked_views, masks = geometric_masking(seq_x)                                      # expects (B, L, C)
        seq_x = seq_x.permute(0, 2, 1)                                                      # (B, C, L)
        # print("SEQ_X Size: ",seq_x.shape)

        # masked_views = torch.from_numpy(masked_views, device = seq_x.device)                    # (B, C, L)
        # masked_views = torch.from_numpy(masked_views).float().to(seq_x.device)
        masked_views = torch.from_numpy(masked_views).float().to(seq_x.device).permute(0,2,1)
        
        # Stack masked views
        inputs = torch.cat([
            torch.stack([seq_x[i]] + [masked_views[i * self.num_masked + j] for j in range(self.num_masked)], dim=0)
            for i in range(B)
        ], dim=0)                                                                           # (B*(M+1), C, L)
        
        inputs = inputs.to(dtype=torch.float32, device=seq_x.device)
        # Encode
        enc_output = self.encoder(inputs)                                                   # (B*(M+1), resnet_out_dim)
        # print("Encoder output shape:", enc_output.shape) 

        # Project
        series_proj = self.projector(enc_output)

        # Similarity matrix
        R = series_wise_similarity(series_proj.mean(dim=1))
        # print("R Size:", R.shape)
        # print("R:", R)
        enc_output = enc_output.unsqueeze(2)
        
        aggregated = point_wise_reconstruction(R, enc_output, num_masked = self.num_masked)  # (B, L, D)
        # print("Aggregation Size (pre squeeze): ", aggregated.shape)
        aggregated = aggregated.squeeze(2)
        # print("Aggregated: ", aggregated)
        # print("Aggregation Size (post squeeze): ", aggregated.shape)                                                     

        # Decode
        reconstructed = self.decoder(aggregated)
        # print("Reconstruction: ", reconstructed)
        # print("Reconstructed Size: ", reconstructed.shape)

        # classification of logits
        pooled = aggregated.mean(dim=1)                                                           # ((B*M+1), D)
        # print("Pooled Size: ", pooled.shape)
        # pooled = pooled_full[:: self.num_masked + 1]                                            # now just (B, d)
        # logits = self.decoder(pooled)  
        # logits = self.cls_head(pooled)                                                          # (B, num_classes)
        total_loss = tot_loss(original, reconstructed, R, self.num_masked, lamb=0.1, t=0.1)
        # print("Total loss:", total_loss.item())
        
        if self.cls_head is not None:
            logits = self.cls_head(pooled)               # (B, num_classes)
        else:
            logits = None

        return total_loss, reconstructed, R, logits


    def training_step(self, seq_x):
        tot_loss, _, _, _ = self(seq_x)
        return tot_loss



# ------------- TRAINING AND FINETUNING -------------
# ------- Epilepsy Data ---------
epi = load_pt(classArgs.epilepsy_dir)

X_pre, y_pre = epi["train"]["samples"], epi["train"]["labels"]        # pre-train on train split
X_fine, y_fine = epi["val"]["samples"], epi["val"]["labels"]          # fine-tune on val split
X_test, y_test = epi["test"]["samples"], epi["test"]["labels"]        # final eval on test split
 
num_classes = int(torch.unique(y_pre).numel())

def make_loader(X, y, batch_size, shuffle):
    X = normalize_data(X) #.permute(0,2,1)
    return dataloader(X, y, batch_size, shuffle = shuffle)

pretrain_loader = make_loader(X_pre,  y_pre, classArgs.cls_pretrain_batch_size, True)
finetune_loader = make_loader(X_fine, y_fine, classArgs.cls_finetune_batch_size, False)
test_loader = make_loader(X_test, y_test, classArgs.cls_finetune_batch_size, False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# C = X_pre.shape[1]
C = X_pre.shape[2]    # this is C



#-------- Pre-Training --------
epi_pretrain_model = SimMTMModelWithResNet(in_channels = C, num_classes = None).to(device)
epi_pretrain_optimizer = optim.Adam(epi_pretrain_model.parameters(), lr = classArgs.cls_pretrain_lr)

for epoch in range(classArgs.cls_pretrain_epochs):
    epi_pretrain_model.train()
    total_loss = 0.0

    for batch_idx, (seq_x,labels) in enumerate(pretrain_loader, start = 1):
        seq_x = seq_x.float().to(device)
        epi_pretrain_optimizer.zero_grad()
        loss = epi_pretrain_model.training_step(seq_x)
        loss.backward()
        epi_pretrain_optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Pretrain | Epoch {epoch+1}/{classArgs.cls_pretrain_epochs} | "f"Batch {batch_idx}/{len(pretrain_loader)} loss={loss:.4f}", end="\r")

    avg_loss = total_loss / len(pretrain_loader)

    print(f"\nPretrain | Epoch {epoch+1} completed — avg loss = {avg_loss:.4f}")

    torch.save(epi_pretrain_model.state_dict(), 'epilepsy_pretrained.pth')


# -------- Fine-Tuning ----------
epi_finetune_model = SimMTMModelWithResNet(in_channels = C , num_classes = num_classes).to(device)
ckpt = torch.load('epilepsy_pretrained.pth')
epi_finetune_model.load_state_dict(ckpt, strict = False)
epi_finetune_optimizer = optim.Adam(epi_finetune_model.parameters(), lr=classArgs.cls_finetune_lr)

print("Epoch | Train Loss | Train Acc% | Val Loss  | Val Acc%")
print("------------------------------------------------------")
for epoch in range(classArgs.cls_finetune_epochs):
    # seq_x = seq_x.float()
    epi_finetune_model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (seq_x, labels) in enumerate(finetune_loader, start = 1):        
        seq_x, labels = seq_x.to(device), labels.to(device)
        epi_finetune_optimizer.zero_grad()
        _, _, _ , logits = epi_finetune_model(seq_x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        epi_finetune_optimizer.step()
        
        # accuracy calculations
        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Finetune T | Epoch {epoch}/{classArgs.cls_finetune_epochs} | " f"Batch {batch_idx}/{len(finetune_loader)} loss={loss:.4f}", end="\r")

    train_loss /= len(finetune_loader)
    train_acc = 100 * train_correct / train_total

    # -- validation -- 
    epi_finetune_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for seq_x, labels in test_loader:
            seq_x, labels = seq_x.float().to(device), labels.to(device)
            _, _, _, logits = epi_finetune_model(seq_x)
            loss = F.cross_entropy(logits, labels)
            
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(test_loader)
    val_acc = 100* val_correct / val_total

    print(f"{epoch:5d} | {train_loss:.4f} | {train_acc:6.2f} | {val_loss:.4f} | {val_acc:6.2f}")

print("\nTraining complete! Saving final checkpoint…")

torch.save(epi_finetune_model.state_dict(), 'epilepsy_finetuned.pth')


# ------- SleepEEG Data Loading --------
sleep = load_pt(classArgs.sleepeeg_dir)

X_pre, y_pre = sleep["train"]["samples"], sleep["train"]["labels"]        # pre-train on train split
X_fine, y_fine = sleep["val"]["samples"], sleep["val"]["labels"]          # fine-tune on val  split
X_test, y_test = sleep["test"]["samples"], sleep["test"]["labels"]        # final eval on test split
 
num_classes = int(torch.unique(y_pre).numel())

def make_loader(X, y, batch_size, shuffle):
    X = normalize_data(X).permute(0,2,1)
    return dataloader(X, y, batch_size, shuffle = shuffle)

pretrain_loader = make_loader(X_pre,  y_pre,  classArgs.cls_pretrain_batch_size, True)
finetune_loader = make_loader(X_fine, y_fine, classArgs.cls_finetune_batch_size, False)
test_loader = make_loader(X_test, y_test, classArgs.cls_finetune_batch_size, False)

if torch.backends.mps.is_available():       # helps run on macbook
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = X_pre.shape[2]


#-------- Pre-Training --------
sleep_pretrain_model = SimMTMModelWithResNet(in_channels = C, num_classes = None).to(device)
sleep_pretrain_optimizer = optim.Adam(sleep_pretrain_model.parameters(), lr = classArgs.cls_pretrain_lr)

for epoch in range(classArgs.cls_pretrain_epochs):
    sleep_pretrain_model.train()
    total_loss = 0.0

    for batch_idx, (seq_x,labels) in enumerate(pretrain_loader, start = 1):
        seq_x = seq_x.float().to(device)
        sleep_pretrain_optimizer.zero_grad()
        loss = sleep_pretrain_model.training_step(seq_x)
        loss.backward()
        sleep_pretrain_optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Pretrain | Epoch {epoch+1}/{classArgs.cls_pretrain_epochs} | "f"Batch {batch_idx}/{len(pretrain_loader)} loss={loss:.4f}", end="\r")

    avg_loss = total_loss / len(pretrain_loader)

    print(f"\nPretrain | Epoch {epoch+1} completed — avg loss = {avg_loss:.4f}")

    torch.save(sleep_pretrain_model.state_dict(), 'sleepeeg_pretrained.pth')


# -------- Fine-Tuning ----------
sleep_finetune_model = SimMTMModelWithResNet(in_channels = C , num_classes = num_classes).to(device)
ckpt = torch.load('sleepeeg_pretrained.pth')
sleep_finetune_model.load_state_dict(ckpt, strict = False)
sleep_finetune_optimizer = optim.Adam(sleep_finetune_model.parameters(), lr = classArgs.cls_finetune_lr)

print("Epoch | Train Loss | Train Acc% | Val Loss  | Val Acc%")
print("------------------------------------------------------")
for epoch in range(classArgs.cls_finetune_epochs):
    # seq_x = seq_x.float()
    sleep_finetune_model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (seq_x, labels) in enumerate(finetune_loader, start = 1):        
        seq_x, labels = seq_x.to(device), labels.to(device)
        sleep_finetune_optimizer.zero_grad()
        _, _, _ , logits = sleep_finetune_model(seq_x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        sleep_finetune_optimizer.step()
        
        # accuracy calculations
        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Finetune T | Epoch {epoch}/{classArgs.cls_finetune_epochs} | " f"Batch {batch_idx}/{len(finetune_loader)} loss={loss:.4f}", end="\r")

    train_loss /= len(finetune_loader)
    train_acc = 100 * train_correct / train_total

    # -- validation -- 
    sleep_finetune_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for seq_x, labels in test_loader:
            seq_x, labels = seq_x.float().to(device), labels.to(device)
            _, _, _, logits = sleep_finetune_model(seq_x)
            loss = F.cross_entropy(logits, labels)
            
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(test_loader)
    val_acc = 100* val_correct / val_total

    print(f"{epoch:5d} | {train_loss:.4f} | {train_acc:6.2f} | {val_loss:.4f} | {val_acc:6.2f}")

print("\nTraining complete! Saving final checkpoint…")

torch.save(sleep_finetune_model.state_dict(), 'sleepeeg_finetuned.pth')



# -- Copied Over Functions ---
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




