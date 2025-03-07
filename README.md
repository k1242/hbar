# hbar

**hbar** is a PyTorch-based library for simulating quantum many-body systems, specifically the 2D Fermi-Hubbard model. It provides tools for Hamiltonian construction and Krylov-based time evolution methods.

<img src="assets/uv.gif" width="500">

## Features
- Efficient Hamiltonian construction for 2D Fermi-Hubbard systems
- Krylov subspace methods for time evolution
- Arbitrary adjacency matrices for system connectivity

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourrepo/hbar.git
cd hbar
pip install -r requirements.txt
```

## Model Description
The system follows the 2D Fermi-Hubbard Hamiltonian:
$$
H = -t\sum_{\langle i, j\rangle} c_i^{\dagger} c_j + u \sum_j n_{j \uparrow} n_{j \downarrow} + v \sum_j V_j (n_{j \uparrow} + n_{j \downarrow})
$$
where:
- \( t \) is the hopping amplitude (default: 1)
- \( u \) is the on-site interaction strength
- \( v \) is the external potential strength
- \( V_j \) is a randomly distributed external potential in the range \([0,1]\)

For different values of \( (u, v) \), localization and thermalization effects can be studied (averaged over multiple noise configurations).

## Additional Resources
For an in-depth discussion on localization and thermalization, see [this essay](https://github.com/k1242/notes_QST/blob/main/MB/ETH2MBL.pdf).

## License
MIT License

