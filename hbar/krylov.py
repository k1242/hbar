import torch

def step(ψ, H, dt, K=10):
    norm = torch.norm(ψ)
    basis = torch.zeros((K, H.N_u * H.N_d), dtype=ψ.dtype, device=ψ.device)
    α = torch.zeros(K, dtype=ψ.dtype, device=ψ.device)
    β = torch.zeros(K, dtype=norm.dtype, device=ψ.device)
    
    basis[0] = ψ.view(-1) / torch.norm(ψ)
    for j in range(K):
        w = H.H(basis[j].view(H.N_u, H.N_d)).view(-1)
        α[j] = torch.vdot(basis[j], w)
        w -= α[j] * basis[j] + (β[j-1] * basis[j-1] if j > 0 else 0)
        β[j] = torch.norm(w)
        if j+1 < K:
            basis[j+1] = w / β[j]
    
    T = torch.diag(α) + torch.diag(β[:-1], 1) + torch.diag(β[:-1], -1)
    return (basis.T @ torch.matrix_exp(-1j * dt * T)[:, 0]).view(H.N_u, H.N_d)
