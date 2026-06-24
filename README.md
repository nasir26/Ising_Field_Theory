# Ising Field Theory Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Numerical simulation of the (1+1)D Ising Field Theory — the quantum field theory
description of the 2D classical Ising model at criticality.

**Author**: Nasir Ali — Centre for Development of Advanced Computing (C-DAC), Noida

## Overview

The 2D classical Ising model at its critical temperature is described by a (1+1)D
Conformal Field Theory (CFT) — the free Majorana fermion theory. This project
simulates the spectrum, correlation functions, and scaling dimensions.

## Contents

- `ising_cft.py` — Exact diagonalisation on finite lattice with periodic BC
- `correlation.py` — Two-point spin/energy correlation functions
- `scaling.py` — Finite-size scaling analysis (extract critical exponents)
- `mps_ising.py` — DMRG/MPS ground state and gap calculation

## Physics

| Quantity | Value |
|----------|-------|
| Central charge | $c = 1/2$ |
| Spin scaling dim. | $\Delta_{\sigma} = 1/8$ |
| Energy scaling dim. | $\Delta_{\epsilon} = 1$ |
| Ising critical temp. | $T_c = 2/\ln(1+\sqrt{2})$ |

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Usage

```python
from ising_cft import IsingChain

chain = IsingChain(L=64, J=1.0, h=1.0)   # h=J is critical point
E, V = chain.diagonalise()
chain.plot_spectrum(num_levels=20)
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nasirali_ising_field_theory,
  author    = {Nasir Ali},
  title     = {Ising Field Theory},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/nasir26/Ising_Field_Theory},
  note      = {Centre for Development of Advanced Computing (C-DAC), Noida, India}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.\
© 2026 Nasir Ali, C-DAC Noida. All rights reserved.
