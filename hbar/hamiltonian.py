import torch
from .utils import state2idx, popcount64, timed
from math import comb

class Hamiltonian:
    """
    Hamiltonian class for a 2D Fermi Hubbard system on arbitrary lattice. 
    This class constructs the Hamiltonian matrix elements for a system with 
    a given adjacency matrix, particle numbers, and interaction parameters.
    
    Parameters:
    - n_u: Number of spin-up (↑) electrons.
    - n_d: Number of spin-down (↓) electrons.
    - adj_mat: Adjacency matrix representing allowed hopping between lattice sites.
    - potential: External potential array.
    - u: On-site interaction strength.
    - t: Hopping amplitude.
    - v: External potential strength.
    - device: Compute device ('cpu' or 'cuda').
    
    Key Methods:
    - H(psi): Apply the Hamiltonian operator to a wavefunction psi.
    - E(psi): Compute the energy expectation value for psi.
    """
    
    def __init__(self, n_u, n_d, adj_mat, potential, u=1.0, t=1.0, v=1.0, device="cpu"):
        self.adj_mat = torch.triu(adj_mat)
        self.L = adj_mat.size(0)  # number of lattice sites
        self.n_u = n_u
        self.n_d = n_d
        self.u = u
        self.t = t
        self.v = v
        self.device = device
        self.potential = potential
        self._build()
    
    def _build(self):
        """Precompute all necessary operators for Hamiltonian."""
        self.N_u = comb(self.L, self.n_u)
        self.N_d = comb(self.L, self.n_d)
        self.N = self.N_u * self.N_d  # total number of basis states
        self.site_idx = torch.arange(self.L, device=self.device)
        self.up = self._gen_combinations(self.n_u)  # ↑ basis
        self.down = self._gen_combinations(self.n_d)  # ↓ basis
        self._gen_T()  # kinetic energy operator
        self._gen_U()  # interaction operator
        self._gen_V()  # potential energy operator

    def _gen_combinations(self, n):
        """Generate all possible n-electron states as bit representations."""
        states = torch.tensor([2**n - 1], device=self.device)  # initial state: first n sites occupied
        while len(states) != comb(self.L, n): # walk with bfs through all states
            states = torch.cat((states, self._hop(states, self.site_idx[:-1], self.site_idx[:-1]+1)[0])).unique()
        return states

    def _gen_U(self):
        """Compute the interaction matrix U diag, counting overlapping electron positions."""
        self._U = popcount64(self.up[:, None] & self.down[None, :])

    def _gen_V(self):
        """Compute the potential energy matrix V diag based on external potential values."""
        pot_u = torch.sum(((self.up[:, None] & (1 << self.site_idx[None, :])) != 0) * self.potential, dim=1)
        pot_d = torch.sum(((self.down[:, None] & (1 << self.site_idx[None, :])) != 0) * self.potential, dim=1)
        self._V = pot_u[:, None] + pot_d[None, :]

    def _gen_hop(self, states):
        """Generate kinetic energy transitions for a given spin configuration."""
        src, dst = self.adj_mat.nonzero(as_tuple=True)  # get allowed hopping transitions
        hopped, valid = self._hop(states, src, dst)  # compute valid hop states
        src_idx = torch.arange(states.size(0), device=self.device).repeat_interleave(self.adj_mat.sum())[valid]
        dst_idx = state2idx(states, hopped)  # map new states to their indices
        mask = torch.bitwise_xor(
            (1 << src.repeat(states.size(0))[valid]) - 1,  
            (1 << dst.repeat(states.size(0))[valid]) - 1
        ) # mask to count fermions between src and dst
        coef = (-1) ** popcount64(hopped & mask)  # compute sign factor based on parity
        return src_idx, dst_idx, coef

    def _gen_T(self):
        """Compute the kinetic energy matrix T using allowed hopping transitions for ↑ and ↓ basis separately."""
        self._T_u = self._gen_hop(self.up)
        self._T_d = self._gen_hop(self.down)

    def _hop(self, states, src, dst):
        """Compute the valid hopping transitions from source to destination for all states."""
        valid = torch.bitwise_and(
            (states[:, None] & (1 << src)).bool(),  # check if src site is occupied
            ~(states[:, None] & (1 << dst)).bool()   # check if dst site is empty
        )
        hopped = torch.bitwise_xor(
            states[:, None],
            (1 << src) | (1 << dst)  # flip the bits for hopping transition
        )[valid]
        return hopped, valid.flatten()

    def upd_V(self, potential):
        """Update the external potential and recompute the potential matrix."""
        self.potential = potential
        self._gen_V()

    def U(self, psi): 
        """Apply the interaction operator U to the wavefunction psi."""
        return self._U * psi
    
    def V(self, psi): 
        """Apply the potential energy operator V to the wavefunction psi."""
        return self._V * psi
    
    def T(self, psi):
        """Apply the kinetic energy operator T to the wavefunction psi."""
        res = torch.zeros_like(psi)
        res.index_add_(0, self._T_u[1], self._T_u[2][:, None] * psi[self._T_u[0], :])  # ↑: src → dst
        res.index_add_(0, self._T_u[0], self._T_u[2][:, None] * psi[self._T_u[1], :])  # ↑: src ← dst
        res.index_add_(1, self._T_d[1], self._T_d[2][None, :] * psi[:, self._T_d[0]])  # ↓: src → dst
        res.index_add_(1, self._T_d[0], self._T_d[2][None, :] * psi[:, self._T_d[1]])  # ↓: src ← dst
        return res

    def H(self, psi):
        """Compute the Hamiltonian applied to the wavefunction psi."""
        return -self.t * self.T(psi) + self.u * self.U(psi) + self.v * self.V(psi)
    
    def E(self, psi):
        """Compute the energy expectation value for wavefunction psi."""
        return torch.vdot(psi.view(-1), self.H(psi).view(-1)).real.item()