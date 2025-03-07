# hbar

**hbar** is a PyTorch-based library for simulating quantum many-body systems, specifically the 2D Fermi-Hubbard model. It provides tools for Hamiltonian construction and Krylov-based time evolution methods. Examples od usage can be found in `notebooks/` directory. Arbitrary adjacency matrices for system connectivity are supported up to 64 lattice sites. 

<img src="assets/uv.mp4" width="400">

## Model Description
The system follows the 2D Fermi-Hubbard Hamiltonian:

![Equation](https://latex.codecogs.com/svg.latex?H%20%3D%20-t%5Csum_%7B%5Clangle%20i%2C%20j%5Crangle%7D%20c_i%5E%7B%5Cdagger%7D%20c_j%20+%20u%20%5Csum_j%20n_%7Bj%20%5Cuparrow%7D%20n_%7Bj%20%5Cdownarrow%7D%20+%20v%20%5Csum_j%20V_j%20(n_%7Bj%20%5Cuparrow%7D%20+%20n_%7Bj%20%5Cdownarrow%7D))

where:
- t is the hopping amplitude (default: 1)
- u is the on-site interaction strength
- v is the external potential strength
- V_j is a randomly distributed external potential in the range [0,1]

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourrepo/hbar.git
cd hbar
pip install -r requirements.txt
```

