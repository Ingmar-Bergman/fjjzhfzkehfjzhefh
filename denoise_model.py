import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond, use_attention=False):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.use_attention = use_attention
        self.dropout = nn.Dropout(0.3)

        # Conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.PReLU(),
            nn.Linear(d_cond, d_cond),
            nn.PReLU(),

        )

        # Time Embedding
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Define MLP layers with residual connections
        self.mlp = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.mlp.append(nn.Linear(input_dim + d_cond, hidden_dim))
            else:
                self.mlp.append(nn.Linear(hidden_dim + d_cond, hidden_dim))

        # Batch Normalization layers
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        num_groups = 8  # or 4, etc., must divide hidden_dim
        self.gn = nn.ModuleList([nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim) 
                         for _ in range(n_layers)])

        # Optional Attention Layer
        if self.use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

        # Activation Function
        self.activation = nn.PReLU()

        # Final Output Layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, cond):
        """
        Forward pass of the denoising model.

        Args:
            x (Tensor): Noisy input data, shape (batch_size, input_dim).
            t (Tensor): Timesteps, shape (batch_size,).
            cond (Tensor): Conditioning information, shape (batch_size, n_cond).

        Returns:
            Tensor: Predicted noise, shape (batch_size, input_dim).
        """
        # Handle conditioning information
        cond = torch.nan_to_num(cond, nan=0.0)  # Replace NaNs with 0.0 or another strategy
        cond = self.cond_mlp(cond)  # Shape: (batch_size, d_cond)

        # Generate time embeddings
        time_emb = self.time_embedding(t)  # Shape: (batch_size, hidden_dim)
        time_emb = self.time_mlp(time_emb)  # Shape: (batch_size, hidden_dim)

        out = x  # Initial input

        for i in range(self.n_layers):
            if i == 0:
                # First layer: concatenate input with conditioning
                out = torch.cat((out, cond), dim=1)
            else:
                # Subsequent layers: concatenate hidden state with conditioning
                out = torch.cat((out, cond), dim=1)

            # Apply MLP layer
            out = self.mlp[i](out)

            # Add time embedding
            out = out + time_emb

            # Apply activation
            out = self.activation(out)

            # Apply Batch Normalization
            # out = self.bn[i](out)
            # Insert a dummy dimension at the end: (N, hidden_dim) -> (N, hidden_dim, 1)
            out = out.unsqueeze(-1)         
            out = self.gn[i](out)           # Now groupnorm sees (N, C=hidden_dim, L=1)
            out = out.squeeze(-1)    

            # Optional Attention
            if self.use_attention:
                # Reshape for attention: (batch_size, seq_length=1, hidden_dim)
                out_reshaped = out.unsqueeze(1)  # Adding seq_length dimension
                attn_output, _ = self.attention(out_reshaped, out_reshaped, out_reshaped)
                out = attn_output.squeeze(1)  # Remove seq_length dimension

        # Final output layer
        out = self.output_layer(out)

        return out


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))