import torch
from .utils import state2idx, popcount64, timed
from math import comb

class Hamiltonian:
    def __init__(self, n_u, n_d, L0, L1, u=1.0, t=1.0, v=1.0, potential=None, device="cpu"):
        self.info = {}
        self.L0 = L0
        self.L1 = L1
        self.L = L0 * L1
        self.n_u = n_u
        self.n_d = n_d
        self.u = u
        self.t = t
        self.v = v
        self.device = device
        self.potential = potential if potential is not None else torch.zeros((self.L,), device=self.device)
        self._build()

    def _build(self):
        self.N_u = comb(self.L, self.n_u)
        self.N_d = comb(self.L, self.n_d)
        self.N = self.N_u * self.N_d
        self._gen_pos()
        self.up = self._gen_combinations(self.n_u, self.L)
        self.down = self._gen_combinations(self.n_d, self.L)
        self._gen_T()
        self._gen_U()
        self._gen_V()

    def _gen_pos(self):
        self.pos0 = torch.arange(self.L-self.L1, device=self.device)
        self.pos1 = torch.arange(self.L, device=self.device)
        self.pos1 = self.pos1[(self.pos1+1) % self.L1 != 0]        
        self.pos_snake = torch.arange(self.L-1, device=self.device) # to generate combinations
        self.pos_full = torch.arange(self.L, device=self.device)

    def _gen_combinations(self, n, L):
        states = torch.tensor([2**n - 1], device=self.device)
        while len(states) != comb(L, n):
            states = torch.cat((states, self._hop(states, self.pos_snake, 1)[0])).unique()
        return states

    def _gen_U(self):
        self._U = popcount64(self.up[:, None] & self.down[None, :])

    def _gen_V(self):
        pot_u = torch.sum(((self.up[:, None] & (1 << self.pos_full[None, :])) != 0) * self.potential, dim=1)
        pot_d = torch.sum(((self.down[:, None] & (1 << self.pos_full[None, :])) != 0) * self.potential, dim=1)
        self._V = pot_u[:, None] + pot_d[None, :]

    def _gen_T(self):
        self._T = (self._gen_Tσ(self.up), self._gen_Tσ(self.down))

    def _hop(self, states, pos, shift):
        valid = (states[:, None] & (1 << pos)).bool() & ~(states[:, None] & (1 << (pos + shift))).bool()
        return (states[:, None] ^ ((1 << pos) | (1 << (pos + shift))))[valid], valid

    def _gen_Tjσ(self, states, pos, shift):
        hopped, valid = self._hop(states, pos, shift)
        src_idx = torch.arange(states.size(0), device=self.device).repeat_interleave(pos.size(0))[valid.flatten()]
        dst_idx = state2idx(states, hopped)
        pos_valid = pos.repeat(states.size(0))[valid.flatten()]    
        mask = ((1 << (pos_valid + shift)) - 1) ^ ((1 << (pos_valid + 1)) - 1)
        coef = (-1)**popcount64(hopped & mask)
        return src_idx, dst_idx, coef

    def _gen_Tσ(self, states):
        src_idx0, dst_idx0, coef0 = self._gen_Tjσ(states, self.pos0, self.L1)
        src_idx1, dst_idx1, coef1 = self._gen_Tjσ(states, self.pos1, 1)
        return torch.cat((src_idx0, src_idx1)), torch.cat((dst_idx0, dst_idx1)), torch.cat((coef0, coef1))

    def U(self, psi): 
        return self._U * psi
    
    def V(self, psi): 
        return self._V * psi
    
    def T(self, psi):
        res = torch.zeros_like(psi)
        res.index_add_(0, self._T[0][1], self._T[0][2][:, None] * psi[self._T[0][0], :]) # up: src → dst
        res.index_add_(0, self._T[0][0], self._T[0][2][:, None] * psi[self._T[0][1], :]) # up: dst → src
        res.index_add_(1, self._T[1][1], self._T[1][2][None, :] * psi[:, self._T[1][0]]) # down: src → dst
        res.index_add_(1, self._T[1][0], self._T[1][2][None, :] * psi[:, self._T[1][1]]) # down: dst → src
        return res

    def H(self, psi):
        return -self.t*self.T(psi) + self.u*self.U(psi) + self.v*self.V(psi)

    def E(self, psi):
        return torch.vdot(psi.view(-1), self.H(psi).view(-1)).real.item()