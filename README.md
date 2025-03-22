# simmtm_rep
EECS 6322 - Reproducing SimMTM, A Simple Pre-Training Framework for Masked Time-Series Modeling

## Project Proposal 💡

### Introduction
SimMTM introduces a novel pre-training strategy for reconstructing time-series data, such as weather patterns, traffic trends, and EEG signals. Traditional masking models excel at reconstructing missing parts using unmasked portions in images and natural language but struggle with time-series data, where direct masking disrupts learning temporal variations.

To address this, SimMTM learns series-wise similarities across multiple masked time series and aggregates point-wise representations to reconstruct the original sequence.

### Model Architecture
The SimMTM architecture consists of an encoder (either a 1DResNet for classification or a transformer for forecasting) that extracts point-wise representations from the input time series. These representations are then processed through point-wise aggregation, which combines multiple masked versions of the same time series. A projection layer (MLP) further transforms the point-wise features into series-wise representations, which are crucial for similarity learning. The series-wise similarity learning module refines these representations, contributing to the constraint loss and feeding its output back into point-wise aggregation to enhance reconstruction. Finally, a decoder (MLP) reconstructs the original time series, and the model is optimized using a combination of reconstruction loss and constraint loss.

### Implementation Plan
The partners will collaborate on implementing point-wise aggregation, projection, series-wise similarity learning, decoder, and loss functions using Pytorch framework. Each partner will implement a different encoder for specific tasks/experiments (classification and forecasting).

### Classification Task
Partner 1 (Claudia Farkas) will implement a 1DResNet encoder for classification with cross-entropy loss.
In-domain classification: Pre-train and fine-tune on the Epilepsy EEG dataset, targeting 94.75% accuracy.
Download dataset, try to have 1dresnet encoder, mlp decoder
Cross-domain classification: Pre-train on SleepEEG, fine-tune on Epilepsy, targeting 95.49% accuracy.

### Forecasting Task
Partner 2 (Kristal Menguc) will implement a vanilla transformer with channel independence for time-series forecasting with L2 loss.
In-domain forecasting: Pre-train and fine-tune on ETTm1, aiming for MSE: 0.348, MAE: 0.385.
Download dataset, try to have vanilla transformer encoder, mlp decoder
Cross-domain forecasting: Pre-train on Weather, fine-tune on ETTm1, aiming for MSE: 0.350, MAE: 0.383.

### Dataset Information
The datasets required for replication are publicly available in the authors' GitHub repository. Here are the sizes of the datasets we will be using:

- Epilepsy EEG (16.4 MB)
- SleepEEG (299.8 MB)
- ETTm1 (10 MB)
- Weather (7.2 MB)

 
 ## File Architecture 🏰
 In this layout:
- **Shared Modules** (masking, projector, decoder, loss functions) are used by both classification and forecasting.
- **Separate Encoders** (1DResNet vs. Transformer) match each partner’s experiments.
- **Notebook Organization**: Each notebook handles a major workflow step (preprocessing, classification, forecasting, evaluation).

Refer to `classification_experiment.ipynb` or `forecasting_experiment.ipynb` to run the respective tasks from pre-training to fine-tuning. For more details on each module, see the docstrings and inline comments within the codebase.

