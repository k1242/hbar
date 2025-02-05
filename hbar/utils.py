import torch
import time

def state2idx(base_states, query_states):
    """Returns indices of query_states in base_states."""
    sorted_idx = torch.argsort(base_states)
    return sorted_idx[torch.searchsorted(base_states[sorted_idx], query_states)]

def popcount64(x):
    """Counts the number of 1 bits (popcount) in each 64-bit integer."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F

def timed(fn):
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        result = fn(self, *args, **kwargs)
        elapsed = time.time() - t0
        if fn.__name__ in self.info:
            self.info[fn.__name__] += elapsed
        else:
            self.info[fn.__name__] = elapsed
        return result
    return wrapper

# transform hamiltonian H to matrix M
def H2M(H):
    def gen_TM():
        (src_u, dst_u, c_u), (src_d, dst_d, c_d) = H._T
        ar_d = torch.arange(H.N_d, device=H.device)
        ar_u = torch.arange(H.N_u, device=H.device)
        x = torch.cat([
            (dst_u[:, None] * H.N_d + ar_d).reshape(-1),
            (src_u[:, None] * H.N_d + ar_d).reshape(-1),
            (ar_u[:, None] * H.N_d + dst_d[None, :]).reshape(-1),
            (ar_u[:, None] * H.N_d + src_d[None, :]).reshape(-1)
        ])
        y = torch.cat([
            (src_u[:, None] * H.N_d + ar_d).reshape(-1),
            (dst_u[:, None] * H.N_d + ar_d).reshape(-1),
            (ar_u[:, None] * H.N_d + src_d[None, :]).reshape(-1),
            (ar_u[:, None] * H.N_d + dst_d[None, :]).reshape(-1)
        ])
        z = torch.cat([
            c_u[:, None].expand(-1, H.N_d).reshape(-1),
            c_u[:, None].expand(-1, H.N_d).reshape(-1),
            c_d[None, :].expand(H.N_u, -1).reshape(-1),
            c_d[None, :].expand(H.N_u, -1).reshape(-1)
        ]).to(torch.complex64)
        return torch.sparse_coo_tensor(torch.stack([x, y]), z, size=(H.N, H.N))

    def gen_diag(values):
        idx = torch.arange(H.N, device=H.device)
        return torch.sparse_coo_tensor(torch.stack([idx, idx]), values.view(-1).to(torch.complex64), size=(H.N, H.N))

    return gen_TM(), gen_diag(H._U), gen_diag(H._V)

# generate proector to ψ_ab based on the hamiltonian H
def H2P(H, mask_a=None):
    mask_a = (1 << (H.L // 2)) - 1 if mask_a is None else mask_a
    mask_b = ((1 << H.L) - 1) ^ mask_a

    ua_idx, ub_idx, da_idx, db_idx = map(
        lambda x: torch.unique(x[0] & x[1], return_inverse=True)[1].flatten(),
        [(H.up, mask_a), (H.up, mask_b), (H.down, mask_a), (H.down, mask_b)]
    )

    row_idx = (ua_idx[:, None] * da_idx.max() + da_idx[None, :]).flatten()
    col_idx = (ub_idx[:, None] * db_idx.max() + db_idx[None, :]).flatten()

    return (row_idx, col_idx)