import os
import torch
from torch.utils.data import TensorDataset, DataLoader

# --- Data Loading ---
def load_pt(dataset_dir):
    '''
    Loads train, validation and test sets form the data/classification/ files
    Args: 
        dataset_dir (str): Path to dataset directory
    Returns: 
        dict: dictionary with keys 'train', 'val' and 'test', each mapping to 'data' and 'label' dictionary
    '''
    # print("Current Working Directory:", os.getcwd())
    data_dict = {}
    for split_file in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, f"{split_file}.pt")
        print("File Path Found: ", file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found!")
        data_dict[split_file] = torch.load(file_path)
    return data_dict

# epilepsy_dir = "../data/epilepsy/"
# sleepeeg_dir = "../data/sleepEEG/"

def load_classification_dataset(epilepsy_dir, sleepeeg_dir):
    print("\nLoading the Epilepsy data set from: ", epilepsy_dir)
    epilepsy_dir = load_pt(epilepsy_dir)
    print("\nLoading the SleepEEG data set from: ", sleepeeg_dir)
    sleepeeg_dir = load_pt(sleepeeg_dir)
    print("\n")

# load_classification_dataset(epilepsy_dir, sleepeeg_dir)


def normalize_data(X):
    '''
    Normalizes each sample in a batch of time serise data
    Args:
        X (torch.Tensor): Input tensor of shape (num_samples, sequence_length, channels)
    Returns:
        torch.Tensor: Normalized data tensor of same shape
    
    '''
    print("Normalizing the Data...")
    mean = X.mean(dim = 1, keepdim = True)
    # std = X.std(dim = 1, keepdim = True)
    # std.clamp(min = 1e-6)
    std  = X.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)

    X_normalized = (X - mean) / std

    print(f"Data Normalized!:\n {X_normalized}, \nSample:" )
    print(X_normalized[0])

    return X_normalized

X = torch.randn(100, 10, 3)                               # num_samples=5, sequence_length=10, channels=3)
normalized_X = normalize_data(X)

def dataloader(X, y, batch_size = 32, shuffle = True):
    '''
    Creates a DataLoader from input data and labels to help train on smaller batches
    Args: 
        X(torch.Tensor): Data tensor of shape (num_samples, sequence_length, channels)
        y (torch.Tensor): Label rensor of shape (num_samples)
        batch_size (int): Batch size
        shuffle (bool): To shuffle the data
    
    '''
    print("Creating DataLoader...")
    assert X.size(0) == y.size(0), f"Size mismatch! X has {X.size(0)} samples, y has {y.size(0)} labels"
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    print(f"Data Loader Created with {len(loader)} batches.")
    return loader
y = torch.randint(0, 3, (100,))
loader = dataloader(X, y, batch_size = 16)