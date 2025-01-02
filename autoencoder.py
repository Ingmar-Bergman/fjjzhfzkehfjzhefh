import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GATv2Conv

# import sagpooling
from torch_geometric.nn import SAGPooling

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,GINEConv, global_mean_pool, global_add_pool, global_max_pool, JumpingKnowledge
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import BatchNorm



# # Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.PReLU()
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


# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, use_prelu=True, weight_sharing=False):
#         super(Decoder, self).__init__()
#         self.n_layers = n_layers
#         self.n_nodes = n_nodes
#         self.use_prelu = use_prelu
#         self.weight_sharing = weight_sharing

#         # Initialize layers
#         layers = []
#         for i in range(n_layers - 1):
#             if weight_sharing and i > 0:
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))  # Shared weight layers
#             else:
#                 in_dim = latent_dim if i == 0 else hidden_dim
#                 layers.append(nn.Linear(in_dim, hidden_dim))
#             if use_prelu:
#                 layers.append(nn.PReLU())  # Learnable activation
#             else:
#                 layers.append(nn.ReLU())  # Fixed activation
#             layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization

#         # Final layer
#         layers.append(nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2))

#         # Sequential MLP
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         # Pass through MLP
#         x = self.mlp(x)

#         # Reshape output for adjacency computation
#         x = torch.reshape(x, (x.size(0), -1, 2))

#         # Gumbel softmax sampling
#         x = F.gumbel_softmax(x, tau=1, hard=True)[:, :, 0]

#         # Build adjacency matrix
#         adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
#         idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
#         adj[:, idx[0], idx[1]] = x

#         # Symmetrize adjacency matrix
#         adj = adj + adj.transpose(1, 2)

        return adj

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.fc0 = nn.Linear(input_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for layer in range(n_layers):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.PReLU(),  # Replace LeakyReLU with learnable PReLU
                        nn.BatchNorm1d(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.PReLU()  # Replace LeakyReLU with learnable PReLU
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        x = self.fc0(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.bns[i](x)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out



# class GIN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
#         super().__init__()
#         self.dropout = dropout
        
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
#                             nn.PReLU(),
#                             nn.BatchNorm1d(hidden_dim),
#                             nn.Linear(hidden_dim, hidden_dim), 
#                             nn.PReLU())
#                             ))                        
#         for layer in range(n_layers-1):
#             self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
#                             nn.PReLU(),
#                             nn.BatchNorm1d(hidden_dim),
#                             nn.Linear(hidden_dim, hidden_dim), 
#                             nn.PReLU())
#                             )) 

#         self.bn = nn.BatchNorm1d(hidden_dim)
#         self.fc = nn.Linear(hidden_dim, latent_dim)

#     def forward(self, data):
#         edge_index = data.edge_index
#         x = data.x

#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = F.dropout(x, self.dropout, training=self.training)

#         out = global_add_pool(x, data.batch)
#         out = self.bn(out)
#         out = self.fc(out)
#         return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, n_max_nodes, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                nn.PReLU(),
                                                nn.BatchNorm1d(hidden_dim),
                                                nn.Linear(hidden_dim, hidden_dim),
                                                nn.PReLU())))
        for _ in range(n_layers - 1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                    nn.PReLU(),
                                                    nn.BatchNorm1d(hidden_dim),
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.PReLU())))

        self.fc1 = nn.Linear(hidden_dim + n_cond + n_max_nodes**2, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, data, adj):
        x, edge_index, batch, stats = data.x, data.edge_index, data.batch, data.stats

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)

        # Concatenate the flattened adjacency matrix with x and stats
        x_cat = torch.cat((x, stats, adj.view(adj.size(0), -1)), dim=1)
        x_temp = self.fc1(x_cat)

        x_temp = F.relu(x_temp, inplace=False)
        x_l = x_temp
        x = self.fc2(x_temp)
        return torch.sigmoid(x), x_l # vérifier l'output



    

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes,n_cond, n_layers_discri):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes) 

        self.discriminator = Discriminator(input_dim, hidden_dim_enc, n_layers_discri, n_cond, n_max_nodes) #tester le nombre de courches
        

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        # mu = F.linear(x_g, self.fc_mu.weight.clone(), self.fc_mu.bias.clone())
        logvar = self.fc_logvar(x_g)
        # logvar = F.linear(x_g, self.fc_logvar.weight.clone(), self.fc_logvar.bias.clone())

        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj, mu, logvar,x_g

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj
    


    def reconstruction_loss(self, data, x, x_hat, z):
        # #Discriminator output for real data
        _, real_features = self.discriminator(data, x.view(-1, self.n_max_nodes * self.n_max_nodes))
        
        # #Discriminator output for reconstructed data
        _, fake_features = self.discriminator(data, x_hat.view(-1, self.n_max_nodes * self.n_max_nodes))

        # Reconstruction loss based on discriminator output probabilities
        feature_recon_loss = F.mse_loss(real_features, fake_features, reduction='mean')

        return feature_recon_loss

    def loss_function(self, data, beta=0.05,gamma = 0.05):
        x_g  = self.encoder(data)
        # mu = F.linear(x_g, self.fc_mu.weight.clone(), self.fc_mu.bias.clone())
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        # logvar = F.linear(x_g, self.fc_logvar.weight.clone(), self.fc_logvar.bias.clone())
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        
        #
        # recon = F.l1_loss(adj, data.A, reduction='mean')
        # C'est le  L discri llike 
        recon = self.reconstruction_loss(data, data.A, adj, x_g)

        # KL divergence loss = PRIOR loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # VAE loss combines reconstruction and KL divergence
        vae_loss = recon + beta*kld

        # GAN loss
        batch_size = data.A.size(0)
        zp = torch.randn(batch_size, self.latent_dim).to(data.A.device)
        xp = self.decoder(zp)
      
        real_output,_ = self.discriminator(data,data.A.view(-1, self.n_max_nodes * self.n_max_nodes))
        fake_output,_ = self.discriminator(data,adj.view(-1, self.n_max_nodes * self.n_max_nodes))
        xp_output,_ = self.discriminator(data,xp.view(-1,self.n_max_nodes * self.n_max_nodes))
        # Discriminator loss
        d_loss = torch.mean(torch.log(real_output + 1e-8) + torch.log(1 - fake_output + 1e-8) + torch.log(1 - xp_output + 1e-8))

        g_loss = -torch.mean(torch.log(fake_output + 1e-8))


        # Combined loss for updating Decoder/Generator, weighted by gamma
        #Decoder loss
        combined_loss = gamma * recon - d_loss


        return vae_loss, combined_loss, d_loss, recon, kld, g_loss

    
