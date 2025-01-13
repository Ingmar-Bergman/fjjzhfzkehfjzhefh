import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


from torch_geometric.nn import GraphNorm

class GraphNormalizedDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, dropout = 0.3):
        super(GraphNormalizedDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers-2)
        ]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.prelu = nn.PReLU()
        self.graph_norm = GraphNorm(hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_nodes=50):
        for i in range(self.n_layers-1):
            x = self.mlp[i](x)
            x = self.graph_norm(x)  # Utilisation de GraphNorm
            x = self.prelu(x)
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + adj.transpose(1, 2)
        return adj




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, GraphNorm
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch

from torch_geometric.loader import DataLoader

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, GATConv):
        if hasattr(module, 'lin') and hasattr(module.lin, 'weight'):
            nn.init.xavier_uniform_(module.lin.weight)
        if hasattr(module, 'att') and module.att is not None:
            nn.init.xavier_uniform_(module.att)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.3,n_heads=1, use_graph_norm=True):
        super(GATEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.prelu_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        # Première couche GATConv
        self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=n_heads, dropout=dropout))
        self.prelu_layers.append(nn.PReLU())
        # Residual pour la première couche : transforme input_dim en hidden_dim * n_heads
        self.residual_layers.append(nn.Linear(input_dim, hidden_dim * n_heads))
        
        # Couches suivantes avec connexions résiduelles
        for _ in range(n_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * n_heads, hidden_dim, heads=n_heads, dropout=dropout))
            if use_graph_norm:
                self.norm = GraphNorm(hidden_dim * n_heads)
            else:
                self.norm = nn.LayerNorm(hidden_dim * n_heads)
            self.prelu_layers.append(nn.PReLU())
            # Residual connections où input_dim == hidden_dim * n_heads
  
            self.residual_layers.append(nn.Identity())
        
        # Global pooling
        self.global_pool = global_add_pool
        
        # # Normalisation
        # if use_graph_norm:
        #     self.norm = GraphNorm(hidden_dim * n_heads)
        # else:
        #     self.norm = nn.LayerNorm(hidden_dim * n_heads)
        
        # Couche finale pour l'espace latent
        self.fc = nn.Linear(hidden_dim * n_heads, latent_dim)
        
        # Taux de dropout
        self.dropout_rate = dropout
        
        # Initialiser les poids
        self.apply(init_weights)
        
    def forward(self, data):
        """
        Forward pass de l'encodeur GAT.

        Args:
            data (Data): Un objet de données PyTorch Geometric contenant x, edge_index, et batch.

        Returns:
            Tensor: Vecteur latent, shape (batch_size, latent_dim).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, prelu, residual in zip(self.convs, self.prelu_layers, self.residual_layers):
            x_input = x
            x = conv(x, edge_index)
            x = self.norm(x)
            x = prelu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # # Appliquer la connexion résiduelle
            # residual_x = residual(x_input)
            # x = x + residual_x  # Maintenant, x et residual_x ont la même forme [batch_size, hidden_dim * n_heads]
        
        x = self.global_pool(x, batch)
        x = self.norm(x)
        z = self.fc(x)
        return z

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def log_normal(z, mu, logvar):
    """
    Compute element-wise log pdf of a Gaussian N(z | mu, exp(logvar)).
    z: (batch_size, latent_dim)
    mu: (batch_size, latent_dim) or (1, latent_dim)
    logvar: (batch_size, latent_dim) or (1, latent_dim)
    Returns: (batch_size) the log probability for each sample in the batch, summed over latent_dim.
    """
    # If necessary, broadcast mu, logvar to match z shape
    c = -0.5 * math.log(2.0 * math.pi)
    return c * z.size(1) - 0.5 * torch.sum(logvar, dim=1) \
           - 0.5 * torch.sum((z - mu)**2 / logvar.exp(), dim=1)

class GaussianMixturePrior(nn.Module):
    """
    A simple Gaussian mixture prior:
    p(z) = sum_{k=1..K} pi_k * N(z | mu_k, exp(logvar_k))
    """
    def __init__(self, n_components, latent_dim):
        super().__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim

        # Mixture weights (logits for numerical stability)
        self.logits = nn.Parameter(torch.zeros(n_components))

        # Means and log-variances for each component
        self.means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.logvars = nn.Parameter(torch.zeros(n_components, latent_dim))

    def forward(self, z):
        """
        Compute log p(z) for the mixture prior, using a stable log-sum-exp trick.
        z : (batch_size, latent_dim)
        Returns: (batch_size) log p(z).
        """
        B = z.size(0)
        K = self.n_components  # Use the correct number of components here

        # Expand z to shape (B, K, D)
        z_expand = z.unsqueeze(1).expand(B, K, self.latent_dim)  # (B, K, D)

        # means and logvars already have the correct shape (K, D)
        # We need to expand them to (B, K, D) for broadcasting
        means_expand = self.means.unsqueeze(0).expand(B, K, self.latent_dim)  # (B, K, D)
        logvars_expand = self.logvars.unsqueeze(0).expand(B, K, self.latent_dim)  # (B, K, D)

        # log p(z | component k)
        comp_log_probs = self.log_normal(z_expand, means_expand, logvars_expand)  # (B, K)

        # mixture weights
        weights = F.softmax(self.logits, dim=0)  # shape (K,)
        log_weights = torch.log(weights + 1e-10)  # (K,)

        # Repeat log_weights for each sample in the batch
        log_weights = log_weights.unsqueeze(0).expand(B, -1)  # (B, K)

        # combine
        # log p(z) = log sum_k [ pi_k * N(z|mu_k, sigma_k) ]
        #          = log_sum_exp( log pi_k + log N_k ), over k

        log_pz = torch.logsumexp(log_weights + comp_log_probs, dim=1)  # shape (B,)
        return log_pz

    def log_normal(self, z, mu, logvar):
      """
      Compute the log probability of a Gaussian distribution with diagonal covariance.
      z : (batch_size, n_components, latent_dim)
      mu : (batch_size, n_components, latent_dim)
      logvar : (batch_size, n_components, latent_dim)
      Returns: (batch_size, n_components) log probability.
      """
      normalization = -0.5 * (torch.log(2 * torch.tensor(math.pi)) + logvar)
      inv_sigma = torch.exp(-logvar)
      tmp = (z - mu)
      return torch.sum(normalization - 0.5 * tmp * tmp * inv_sigma, dim=2)



# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, dropout):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # self.encoder = GATEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout)
        self.encoder = GATEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout)

        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = GraphNormalizedDecoder(latent_dim, hidden_dim_dec, n_layers_dec,n_max_nodes, dropout)

        self.gmm_prior = GaussianMixturePrior(n_components=2, latent_dim=latent_dim)

    # Beta annealing parameters
        self.beta_start = 0
        self.beta_end = 0.00005
        self.beta_n_steps = 200  # Number of steps (epochs or iterations) for annealing
        self.beta_annealing_type = 'linear'
        self.beta_step_size = (self.beta_end - self.beta_start) / self.beta_n_steps
        self.current_beta = 0
        self.current_step = 0

    def update_beta(self):
        if self.beta_annealing_type == 'linear':
            self.current_beta = min(self.beta_end, self.beta_start + self.current_step * self.beta_step_size)
        elif self.beta_annealing_type == 'cyclical':
            cycle_length = self.beta_n_steps // 4  # Example: 4 cycles
            self.current_beta = self.beta_end * (self.current_step % cycle_length) / cycle_length
        elif self.beta_annealing_type == 'monotonic':
            if self.current_beta < self.beta_end:
                self.current_beta += self.beta_step_size
                self.current_beta = min(self.current_beta, self.beta_end)
        else:
            raise ValueError("Invalid beta_annealing_type. Choose from 'linear', 'cyclical', 'monotonic'.")
        self.current_step += 1


    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)          # (batch_size, latent_dim)
        logvar = self.fc_logvar(x_g)  # (batch_size, latent_dim)
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        z = self.reparameterize(mu, logvar)
        return z

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std) * eps_scale
            return mu + eps * std
        else:
            return mu

    def decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)
        return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj
    
    def log_q_z_x(self, z, mu, logvar):
        """
        Computes log q(z|x) under the diagonal Gaussian with parameters (mu, logvar).
        We sum over the latent dimension, returning a shape (batch_size,) 
        """
        return log_normal(z, mu, logvar)
    
    def loss_function(self, data, beta=0.05, n_samples=30): #add n_samples argument for MC estimation
        """
        Modified loss to incorporate a mixture prior in the KLD term,
        using Monte Carlo estimation for the KL divergence.
        """
        # 1) Encode
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)           # (batch_size, latent_dim)
        logvar = self.fc_logvar(x_g)   # (batch_size, latent_dim)

        # 2) Reparameterize -> sample z (multiple times for MC estimation)
        z_samples = [self.reparameterize(mu, logvar) for _ in range(n_samples)]

        # 3) Decode -> adjacency reconstruction
        recon_losses = []
        for z in z_samples:
            adj = self.decoder(z)
            recon_losses.append(F.l1_loss(adj, data.A, reduction='sum') / (data.A.shape[0] * self.n_max_nodes * self.n_max_nodes))
        recon = torch.stack(recon_losses).mean()  # Average reconstruction loss over samples

        # 4 & 5) Compute log q(z|x) and log p(z) for each sample
        kl_values = []
        for z in z_samples:
            log_qzx = self.log_q_z_x(z, mu, logvar)  # shape (batch_size,)
            log_pz = self.gmm_prior(z)  # shape (batch_size,)
            kl_values.append(log_qzx - log_pz)

        # 6) KLD ~ E[ log q(z|x) - log p(z) ] using Monte Carlo estimation
        kl_gmm = torch.stack(kl_values).mean()  # Average KL over samples

        # 7) Final VAE loss = recon + beta * KL
        loss = recon + self.current_beta * kl_gmm
        self.update_beta()

        return loss, recon, kl_gmm



# # Variational Autoencoder
# class VariationalAutoEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, dropout, K = 10):
#         super(VariationalAutoEncoder, self).__init__()
#         self.n_max_nodes = n_max_nodes
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         # self.encoder = GATEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout)
#         self.encoder = GATEncoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, dropout)
#         self.K = K
#         self.pseudo_inputs = nn.Parameter(torch.randn(K, input_dim))

#         self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
#         self.decoder = GraphNormalizedDecoder(latent_dim, hidden_dim_dec, n_layers_dec,n_max_nodes, dropout)

#     # # Beta annealing parameters
#     #     self.beta_start = 0
#     #     self.beta_end = 0.05
#     #     self.beta_n_steps = 50  # Number of steps (epochs or iterations) for annealing
#     #     self.beta_annealing_type = 'linear'
#     #     self.beta_step_size = (self.beta_end - self.beta_start) / self.beta_n_steps
#     #     self.current_beta = 0
#     #     self.current_step = 0

#     # def update_beta(self):
#     #     if self.beta_annealing_type == 'linear':
#     #         self.current_beta = min(self.beta_end, self.beta_start + self.current_step * self.beta_step_size)
#     #     elif self.beta_annealing_type == 'cyclical':
#     #         cycle_length = self.beta_n_steps // 4  # Example: 4 cycles
#     #         self.current_beta = self.beta_end * (self.current_step % cycle_length) / cycle_length
#     #     elif self.beta_annealing_type == 'monotonic':
#     #         if self.current_beta < self.beta_end:
#     #             self.current_beta += self.beta_step_size
#     #             self.current_beta = min(self.current_beta, self.beta_end)
#     #     else:
#     #         raise ValueError("Invalid beta_annealing_type. Choose from 'linear', 'cyclical', 'monotonic'.")
#     #     self.current_step += 1


#     def forward(self, data):
#         x_g = self.encoder(data)
#         mu = self.fc_mu(x_g)
#         logvar = self.fc_logvar(x_g)
#         x_g = self.reparameterize(mu, logvar)
#         adj = self.decoder(x_g)
#         return adj

#     def encode(self, data):
#         x_g = self.encoder(data)
#         mu = self.fc_mu(x_g)
#         logvar = self.fc_logvar(x_g)
#         x_g = self.reparameterize(mu, logvar)
#         return x_g

#     def reparameterize(self, mu, logvar, eps_scale=1.):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = torch.randn_like(std) * eps_scale
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def decode(self, mu, logvar):
#        x_g = self.reparameterize(mu, logvar)
#        adj = self.decoder(x_g)
#        return adj


#     def loss_function(self, data, beta=0.05):
#         x_g  = self.encoder(data)
#         mu = self.fc_mu(x_g)
#         logvar = self.fc_logvar(x_g)
#         x_g = self.reparameterize(mu, logvar)
#         adj = self.decoder(x_g)
        
#         recon = F.binary_cross_entropy(adj, data.A, reduction='mean')
#         kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         loss = recon + beta*kld
#         # self.update_beta()

#         return loss, recon, kld


    
class AttentionMultiHeadGraphNormalizedDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, num_heads=1):
        super(AttentionMultiHeadGraphNormalizedDecoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

        # Couche d'attention multi-têtes
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)]
        graph_norm_layers = [GraphNorm(hidden_dim)]
        prelu_layers = [nn.PReLU()]

        for _ in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            graph_norm_layers.append(GraphNorm(hidden_dim))
            prelu_layers.append(nn.PReLU())

        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.graph_norm = nn.ModuleList(graph_norm_layers)
        self.prelu = nn.ModuleList(prelu_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, num_nodes=50):
        # Appliquer une transformation initiale
        x = self.mlp[0](x)
        x = self.graph_norm[0](x)
        x = self.prelu[0](x)
        # Ajouter une dimension séquentielle pour l'attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        attn_output, _ = self.attention(x, x, x)  # (batch_size, 1, hidden_dim)
        x = attn_output.squeeze(1)  # (batch_size, hidden_dim)

        for i in range(1, self.n_layers - 1):
            x = self.mlp[i](x)
            x = self.graph_norm[i](x)
            x = self.prelu[i](x)
        
        x = self.mlp[self.n_layers - 1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + adj.transpose(1, 2)
        return adj


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm

class AttentionMultiHeadGraphNormalizedDecoderDrop(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, dropout=0,num_heads=4):
        super(AttentionMultiHeadGraphNormalizedDecoderDrop, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

        # Couche d'attention multi-têtes
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Définir les couches MLP avec Dropout
        mlp_layers = [nn.Linear(latent_dim, hidden_dim)]
        graph_norm_layers = [GraphNorm(hidden_dim)]
        prelu_layers = [nn.PReLU()]
        # dropout_layers = [nn.Dropout(p=dropout)]

        for _ in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            graph_norm_layers.append(GraphNorm(hidden_dim))
            prelu_layers.append(nn.PReLU())
            # dropout_layers.append(nn.Dropout(p=dropout))

        mlp_layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.graph_norm = nn.ModuleList(graph_norm_layers)
        self.prelu = nn.ModuleList(prelu_layers)
        # self.dropout = nn.ModuleList(dropout_layers)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        # Initialisation des poids (optionnel mais recommandé)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GraphNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, num_nodes=50):
        # Appliquer la première couche linéaire
        x = self.mlp[0](x)  # (batch_size, hidden_dim)
        x = self.graph_norm[0](x)
        x = self.prelu[0](x)
        # x = self.dropout[0](x)

        # Ajouter une dimension séquentielle pour l'attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        attn_output, _ = self.attention(x, x, x)  # (batch_size, 1, hidden_dim)
        x = attn_output.squeeze(1)  # (batch_size, hidden_dim)

        # Appliquer les couches intermédiaires avec Dropout
        for i in range(1, self.n_layers - 1):
            x = self.mlp[i](x)
            x = self.graph_norm[i](x)
            x = self.prelu[i](x)
            # x = self.dropout[i](x)

        # Couche de sortie
        x = self.mlp[self.n_layers - 1](x)  # (batch_size, 2 * n_nodes * (n_nodes - 1) // 2)
        x = self.dropout(x)
        x = torch.reshape(x, (x.size(0), -1, 2))  # (batch_size, num_edges, 2)
        x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]  # (batch_size, num_edges)

        # Construire la matrice d'adjacence
        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = x
        adj = adj + adj.transpose(1, 2)

        return adj

