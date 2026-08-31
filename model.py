import torch
import torch.nn as nn
import torch.nn.functional as F

def get_deterministic_grid_knn(grid_size, device, k=8, dilation=1):
    """Cached O(1) deterministic 8-neighbor grid stencil."""
    N = grid_size * grid_size
    idx = torch.arange(N, device=device).view(1, 1, grid_size, grid_size)
    padded = F.pad(idx.float(), (dilation, dilation, dilation, dilation), mode='reflect').long().squeeze(0).squeeze(0)
    step = dilation
    neighbors = [
        padded[:-2*step, step:-step], padded[2*step:, step:-step],
        padded[step:-step, :-2*step], padded[step:-step, 2*step:],
        padded[:-2*step, :-2*step], padded[:-2*step, 2*step:],
        padded[2*step:, :-2*step], padded[2*step:, 2*step:]
    ]
    knn_idx = torch.stack(neighbors[:k], dim=-1).reshape(N, k)
    return knn_idx.unsqueeze(0)

class ISNBlock(nn.Module):
    def __init__(self, hidden_dim=384):
        super().__init__()
        # h_i, h_j, mag, cos_phase, sin_phase, z_depth
        edge_in_dim = 2 * hidden_dim + 4  
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, h, coords, e1, e2, neighbor_idx, sigma, z_depth):
        B, N, C = h.shape
        K = neighbor_idx.shape[-1]

        idx_flat = neighbor_idx.reshape(B, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B,N,K,2)
        h_j = torch.gather(h, 1, idx_flat.unsqueeze(-1).expand(-1,-1,C)).view(B,N,K,C)

        h_i = h.unsqueeze(2).expand(-1,-1,K,-1)
        coords_i = coords.unsqueeze(2).expand(-1,-1,K,-1)
        sigma_i = sigma.unsqueeze(2).expand(-1,-1,K,-1)

        delta_x = coords_j - coords_i
        dx_local = torch.sum(delta_x * e1.unsqueeze(2), dim=-1, keepdim=True)
        dy_local = torch.sum(delta_x * e2.unsqueeze(2), dim=-1, keepdim=True)

        mag = torch.sqrt(dx_local**2 + dy_local**2) / (sigma_i + 1e-8)
        phase = torch.atan2(dy_local, dx_local)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        z_depth_i = z_depth.unsqueeze(2).expand(-1,-1,K,-1)

        # Complete edge feature set matching the sealed paper
        edge_input = torch.cat([h_i, h_j, mag, cos_phase, sin_phase, z_depth_i], dim=-1)
        messages = self.edge_mlp(edge_input)
        agg = messages.mean(dim=2)

        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)
        return h + self.res_scale * self.norm(h_new)

class AlienXOperator(nn.Module):
    def __init__(self, hidden_dim=384, L=4, alpha=1.0, sigma_0=0.1):
        super().__init__()
        self.alpha = alpha
        self.sigma_0 = sigma_0
        # Input: k, grad_k_mag, z_depth
        self.input_proj = nn.Linear(3, hidden_dim) 
        self.blocks = nn.ModuleList([ISNBlock(hidden_dim) for _ in range(L)])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, k, grad_k_mag, grad_k_vec, coords, neighbor_idx):
        B, N, _ = coords.shape
        
        # 1. Compute local dispersion (sigma)
        idx_flat = neighbor_idx.reshape(1, -1)
        coords_j = torch.gather(coords, 1, idx_flat.unsqueeze(-1).expand(-1,-1,2)).view(B,N,-1,2)
        coords_i = coords.unsqueeze(2)
        dists = torch.norm(coords_j - coords_i, dim=-1)
        sigma = dists.mean(dim=-1, keepdim=True) # (B, N, 1)
        
        # 2. Scale-space depth (z_depth)
        z_depth = self.alpha * torch.log(sigma / self.sigma_0 + 1e-8) # (B, N, 1)
        
        # 3. Local SO(2) Frame from permeability gradient
        e1 = grad_k_vec / (torch.norm(grad_k_vec, dim=-1, keepdim=True) + 1e-8)
        e2 = torch.stack([-e1[..., 1], e1[..., 0]], dim=-1)
        
        # 4. Input projection
        h = self.input_proj(torch.cat([k.unsqueeze(-1), grad_k_mag.unsqueeze(-1), z_depth], dim=-1))
        
        # 5. ISN Message Passing
        for block in self.blocks:
            h = block(h, coords, e1, e2, neighbor_idx.expand(B, -1, -1), sigma, z_depth)
            
        # 6. Output pressure field
        return self.output_proj(h).squeeze(-1)