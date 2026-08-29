# PoroViscoFEniCS

A generalized poro-viscoelastic finite element solver built on [FEniCS](https://fenicsproject.org/).

## Overview

PoroViscoFEniCS solves coupled poro-viscoelasticity problems in fluid-saturated porous media using a mixed finite element formulation:

- **Pressure**: Discontinuous Galerkin (DG, degree 0)
- **Flux**: Brezzi-Douglas-Marini (BDM, degree 1)
- **Displacement**: Continuous Galerkin (CG, degree 1)

Viscoelastic creep is modeled via the Generalized Maxwell (Wiechert) model with semi-implicit backward Euler integration.

## Requirements

- FEniCS 2019.x (legacy)
- NumPy, Matplotlib (for post-processing)

## Quick Start

```python
from PoroViscoFEniCS import *
# Configure parameters, mesh, boundary conditions
# See examples/ for complete usage
```

## Citation

If you use this code, please cite:

> Giannopoulou, O. (2026). PoroViscoFEniCS: A generalized poro-viscoelastic model using FEniCS. *Computers & Geosciences*.

## License

MIT License