from itertools import product
import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sam import SAM

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset


from torch.utils.data import Subset
np.random.seed(13)

# Define the hyperparameter space
param_grid = {
    'lr': [1e-3],
    'hidden_dim_encoder': [32, 64],
    'hidden_dim_decoder': [128],
    'n_layers_encoder': [2],
    'n_layers_decoder': [3],
    'dropout': [0.5],
    'latent_dim': [16]
}

# Optionally, you can limit the number of combinations to manage computational resources
# For example, select a subset or use random sampling



def train_autoencoder(params, train_loader, val_loader, device,save_dir='autoencoder_models'):
    """
    Initializes, trains, and evaluates the autoencoder based on the provided hyperparameters.
    
    Args:
        params (dict): Dictionary containing hyperparameters.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the model on.
        
    Returns:
        float: Best validation loss achieved.
    """
    # Unpack hyperparameters
    lr = params['lr']
    hidden_dim_encoder = params['hidden_dim_encoder']
    hidden_dim_decoder = params['hidden_dim_decoder']
    n_layers_encoder = params['n_layers_encoder']
    n_layers_decoder = params['n_layers_decoder']
    dropout_start = params['dropout_start']
    dropout_increase = params['dropout_increase']
    latent_dim = params['latent_dim']
    
    # Initialize the autoencoder
    autoencoder = VariationalAutoEncoder(
        input_dim=7+1,  # Adjust based on your data
        hidden_dim_enc=hidden_dim_encoder,
        hidden_dim_dec=hidden_dim_decoder,
        latent_dim=latent_dim,
        n_layers_enc=n_layers_encoder,
        n_layers_dec=n_layers_decoder,
        n_max_nodes=50,dropout_start=dropout_start, dropout_increase = dropout_increase
    ).to(device)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    
    best_val_loss = np.inf
    
    for epoch in range(1, 150 + 1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        train_loss_all_recon = 0
        train_loss_all_kld = 0
        cnt_train = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss, recon, kld = autoencoder.loss_function(data)
            train_loss_all_recon += recon.item()
            train_loss_all_kld += kld.item()
            cnt_train += 1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch) + 1
            optimizer.step()

        # Evaluation on validation set
        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0
        val_loss_all_recon = 0
        val_loss_all_kld = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                loss, recon, kld = autoencoder.loss_function(data)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
                val_loss_all += loss.item()
                cnt_val += 1
                val_count += torch.max(data.batch) + 1

        # Logging
        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(
                f'{dt_t} Epoch: {epoch:04d}, '
                f'Train Loss: {train_loss_all / cnt_train:.5f}, '
                f'Train Reconstruction Loss: {train_loss_all_recon / cnt_train:.2f}, '
                f'Train KLD Loss: {train_loss_all_kld / cnt_train:.2f}, '
                f'Val Loss: {val_loss_all / cnt_val:.5f}, '
                f'Val Reconstruction Loss: {val_loss_all_recon / cnt_val:.2f}, '
                f'Val KLD Loss: {val_loss_all_kld / cnt_val:.2f}'
            )

        # Scheduler step
        scheduler.step()

        # Save the best model
        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'autoencoder_best.pth.tar')

    # Load the best model
    checkpoint = torch.load('autoencoder_best.pth.tar', map_location=device)
    autoencoder.load_state_dict(checkpoint['state_dict'])
    
    return best_val_loss


def grid_search(param_grid, train_loader, val_loader, device, n_repeats=3, save_dir='autoencoder_models'):
    """
    Performs grid search over the specified hyperparameter grid with multiple repeats per combination.
    
    Args:
        param_grid (dict): Dictionary where keys are hyperparameter names and values are lists of possible values.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the model on.
        n_repeats (int): Number of runs per hyperparameter combination.
        save_dir (str): Directory to save model checkpoints.
        
    Returns:
        pd.DataFrame: DataFrame containing hyperparameters, mean validation loss, and standard deviation.
    """
    results = []
    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(product(*values))
    
    total_combinations = len(param_combinations)
    print(f"Total hyperparameter combinations to evaluate: {total_combinations}")
    print(f"Number of repeats per combination: {n_repeats}")
    print(f"Total runs: {total_combinations * n_repeats}")
    
    for idx, combination in enumerate(param_combinations, 1):
        params = dict(zip(keys, combination))
        print(f"\nEvaluating combination {idx}/{total_combinations}: {params}")
        
        val_losses = []
        model_paths = []
        
        for repeat in range(1, n_repeats + 1):
            seed = 42 + repeat  # Example: different seed for each repeat
            print(f"  Run {repeat}/{n_repeats} with seed {seed}")
            val_loss = train_autoencoder(params, train_loader, val_loader, device, save_dir)
            val_losses.append(val_loss)
            print(f"    Run {repeat} completed with Validation Loss: {val_loss:.5f}")
        
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        
        result = params.copy()
        result['mean_val_loss'] = mean_val_loss
        result['std_val_loss'] = std_val_loss
        results.append(result)
        
        print(f"  Combination {idx} completed with Mean Val Loss: {mean_val_loss:.5f} ± {std_val_loss:.5f}")
    
    # Convert results to a DataFrame for easier analysis
    df_results = pd.DataFrame(results)
    return df_results



import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sam import SAM

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset

from torch.utils.data import Subset
from itertools import product
import pandas as pd
import multiprocessing

parser = argparse.ArgumentParser(description='GridSearch for Hyperparameter Optimization')

# Define hyperparameters as before
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")
parser.add_argument('--epochs-autoencoder', type=int, default=150, help="Number of training epochs for the autoencoder (default: 200)")
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")
parser.add_argument('--latent-dim', type=int, default=32, help="Dimensionality of the latent space in the autoencoder (default: 32)")
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")
parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")
parser.add_argument('--spectral-emb-dim', type=int, default=7, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")
parser.add_argument('--epochs-denoise', type=int, default=300, help="Number of training epochs for the denoising model (default: 100)")
parser.add_argument('--timesteps', type=int, default=700, help="Number of timesteps for the diffusion (default: 500)")
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")
parser.add_argument('--train-autoencoder', action='store_false', default=True, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Preprocess datasets
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)

# Initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# param_grid = {
#     'lr': [1e-4, 1e-3, 1e-2],
#     'hidden_dim_encoder': [32, 64],
#     'hidden_dim_decoder': [128, 256],
#     'n_layers_encoder': [2, 3],
#     'n_layers_decoder': [3, 4],
#     'dropout': [0.2, 0.3],
#     'latent_dim': [16, 32]
# }

param_grid = {
    'lr': [1e-3],
    'hidden_dim_encoder': [64],
    'hidden_dim_decoder': [256],
    'n_layers_encoder': [2],
    'n_layers_decoder': [3],
    'dropout_start': [0.1,0.15,0.2],
    'dropout_increase': [0.1,0.15],
    'latent_dim': [32]
}
# Perform grid search
grid_search_results = grid_search(param_grid, train_loader, val_loader, device)

# Save and analyze results
df_results = pd.DataFrame(grid_search_results)
df_results.to_csv("grid_search_results.csv", index=False)

# Find the best result
best_result = df_results.loc[df_results['mean_val_loss'].idxmin()]
print("\nBest Hyperparameter Combination:")
print(best_result)