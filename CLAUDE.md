# CLAUDE.md — AI Assistant Guide

This file provides context for AI assistants (Claude and others) working in this repository.

---

## Project Overview

This repository implements a **topology-optimized 1×4 optical power splitter** for silicon photonics. The goal is to design a broadband, low-loss, compact photonic device that splits one input waveguide signal equally into four output waveguides.

**Key targets:**
- Bandwidth: ≥250 nm (covering telecom O/C bands, 1310/1550 nm)
- Insertion loss: <0.3 dB
- Splitting uniformity: ±0.5 dB imbalance
- Footprint: 3×3 μm²
- Fabrication tolerance: ±10 nm feature variation

**Platform:** Silicon-on-Insulator (SOI), 220 nm Si core, SiO₂ cladding.

**Method:** Density-based topology optimization via the adjoint method, with Tidy3D as the FDTD simulation backend.

---

## Repository Structure

```
srtp-1x4-splitter-optimization/
├── src/srtp_splitter/          # Core library (installable package)
│   ├── __init__.py             # Exports and version (1.0.0)
│   ├── optimizer.py            # Main optimizer class
│   ├── manufacturing.py        # Manufacturing constraints & validation
│   └── utils.py                # Initialization, convergence, metrics
├── examples/
│   ├── basic.py                # Minimal working example
│   └── advanced_multi_wavelength.py  # Multi-wavelength with manufacturing
├── tests/
│   └── test_optimizer.py       # Unit tests (pytest)
├── docs/
│   ├── quickstart.md           # Installation and usage guide
│   └── reports/                # Technical research reports
│       ├── final.md            # Project completion report
│       ├── implementation-strategy.md
│       ├── research-multi-wavelength.md
│       ├── phase1-v2.md
│       ├── fio-decision.md
│       └── phase1.md
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview (Chinese)
└── CLAUDE.md                   # This file
```

---

## Core Architecture

### `SymmetricSplitterOptimizer` (`src/srtp_splitter/optimizer.py`)

The primary optimization engine. Key design decisions:

**4-fold symmetry acceleration**: Only 1/4 of the design domain is optimized (15×15 = 225 variables instead of 30×30 = 900). The full structure is reconstructed by mirroring:
- Quadrant 1: original
- Quadrant 2: flip along x-axis
- Quadrant 3: flip along y-axis
- Quadrant 4: flip both axes

This gives a 4× speedup in both memory and gradient computation.

**Progressive Heaviside projection**: Beta (projection sharpness) increases 1→32 over the optimization run, gradually pushing gray densities toward binary (0/1). Formula:
```
H(x) = (tanh(β·η) + tanh(β·(x-η))) / (tanh(β·η) + tanh(β·(1-η)))
```

**Multi-wavelength objective**: Simultaneously optimizes across 11–21 wavelength points (1.45–1.65 μm). Uses worst-case (max) across wavelengths to produce a conservative wideband design:
```
objective = max(objectives_per_wavelength)
per_wavelength = -transmission + 0.5·uniformity + 0.3·splitting_error
```

**SIMP material interpolation**: Intermediate densities get permittivity:
```
ε = ε_clad + x^p · (ε_wg - ε_clad)    where p=3
ε_Si = 3.45², ε_SiO₂ = 1.45²
```

**Critical limitation (as of current state)**: The Tidy3D simulation calls in `make_structure()` and `calculate_objective()` are **placeholders**. The gradient computation also uses random noise instead of true adjoint derivatives. The framework is complete but not yet wired to real physics.

---

### `ManufacturingConstraints` (`src/srtp_splitter/manufacturing.py`)

Enforces physical realizability:

| Constraint | Default | Purpose |
|------------|---------|---------|
| Min feature size | 80 nm | E-beam lithography limit |
| Grid resolution | 20 nm/pixel | Simulation accuracy |
| Filter radius | 4 pixels (80 nm) | Computed as min_feature/grid_size |
| Binarization threshold | η = 0.5 | Heaviside midpoint |

**Pipeline** (called once per iteration via `apply_all_constraints()`):
1. Density filter (Gaussian smoothing via `scipy.ndimage.gaussian_filter`)
2. Heaviside projection (tanh-based binarization)
3. Final binarize at 0.5 threshold

**Validation** (`validate_manufacturability()`):
- Measures minimum feature sizes via skeletonization (`skimage.morphology.skeletonize`)
- Checks input-output port connectivity via binary morphology
- Returns pass/fail with reason string

---

### `utils.py` — Supporting Tools

| Class | Purpose |
|-------|---------|
| `InitializationStrategies` | Three init methods: `random_uniform` (0.3–0.7), `constant(0.5)`, `radial_gradient` |
| `LearningRateSchedules` | `constant()` or `cosine_annealing()` (recommended) |
| `ConvergenceMonitor` | Early stopping with patience=20, min_delta=1e-6 |
| `ObjectiveFunctions` | `calculate_bandwidth()` and `calculate_imbalance()` metrics |

---

## Physics Grid Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Design region | 3.0 × 3.0 μm | Physical size |
| Resolution | 0.1 μm/pixel | 20 nm manufacturing, 100 nm simulation |
| Full grid | 30 × 30 = 900 pixels | `nx = int(3.0/0.1) = 30` |
| Optimized (symmetric) | 15 × 15 = 225 pixels | Quarter domain |
| Wavelengths | 1.45–1.65 μm | 11–21 points depending on config |

These are set in `SymmetricSplitterOptimizer.__init__()`. Do not change `resolution` without updating `manufacturing.py` grid parameters accordingly.

---

## Development Workflows

### Running Tests

```bash
pytest tests/
```

Tests cover: initialization strategies, symmetry expansion, manufacturing filter shapes, bandwidth/imbalance metrics. No integration tests with real Tidy3D exist yet.

### Running Examples

```bash
python examples/basic.py
python examples/advanced_multi_wavelength.py
```

Both examples currently run with placeholder physics and will print mock convergence data.

### Code Quality

```bash
black src/ tests/ examples/    # Auto-format
flake8 src/ tests/ examples/   # Lint check
```

Black and flake8 are the enforced formatters. Do not introduce style inconsistencies.

### Installing the Package

```bash
pip install -r requirements.txt
pip install -e .               # If setup.py/pyproject.toml is added
```

Currently the package must be used from the repo root (no pyproject.toml yet).

---

## Code Conventions

### Naming
- **Classes**: PascalCase (`SymmetricSplitterOptimizer`, `ManufacturingConstraints`)
- **Methods and variables**: snake_case (`expand_symmetry`, `apply_filter`, `params_quarter`)
- **Physical quantities**: Follow physics notation — use Greek letters as variable names (`eta`, `beta`, `rmin`, `epsilon`)
- **Grid dimensions**: Use `nx`, `ny` (integers) consistently
- **Density fields**: NumPy arrays of shape `(nx, ny)` with values in `[0, 1]`

### Type Hints
All public methods have complete type annotations. Use `np.ndarray` for density/structure arrays, not generic `Any`. Include physical units in docstrings.

### Docstrings
Google/NumPy style. Include `Args:`, `Returns:`, and units where relevant. Example from `optimizer.py`:
```python
def apply_filter(self, params: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Apply Gaussian density filter to prevent checkerboard artifacts.
    
    Args:
        params: Density field array of shape (nx, ny), values in [0, 1]
        radius: Filter radius in pixels (default 2.0 ≈ 0.2 μm)
    
    Returns:
        Filtered density array, same shape as input
    """
```

### Density Fields
Always clip density arrays to `[0, 1]` after gradient updates:
```python
self.params = np.clip(self.params, 0, 1)
```

Never allow densities outside this range — SIMP interpolation and Heaviside projection assume valid domain.

### Imports
Standard order: stdlib → numpy/scipy → domain libs (tidy3d, jax) → local. No wildcard imports.

---

## Key Integration Points for Future Work

### Connecting Real Tidy3D Simulation

The two methods that need real implementation are in `optimizer.py`:

1. **`make_structure(params)`** — Currently returns a mock `tidy3d.Structure`. Should build an actual `td.Structure` with a `PermittivityMonitor` and custom permittivity from the SIMP-interpolated density field.

2. **`calculate_objective(sim_data, wavelengths)`** — Currently returns `(0.24, 0.24, 0.25, 0.25)` mock powers. Should extract actual field amplitudes from the Tidy3D simulation data at the four output ports.

3. **Adjoint gradient** — In `optimize()`, the gradient is currently `np.random.randn(*self.params.shape) * 0.01`. This must be replaced with `sim_data.grad` from a Tidy3D adjoint simulation run.

### JAX Integration

`jax` and `jaxlib` are in requirements but not yet used in the source. The intended path is:
- Convert density arrays to JAX arrays (`jnp.array`)
- Use `jax.grad` or Tidy3D's JAX-based adjoint for gradient computation
- Apply `@jax.jit` to hot loops once Tidy3D integration is stable

---

## Research Context

The `docs/reports/` directory contains the full research history. Key decisions documented there:

- **`fio-decision.md`**: Chose to focus on 1310/1550 nm (O+C bands, 300 nm bandwidth) first before expanding to 1064/980 nm
- **`phase1-v2.md`**: Framework produced 4.56 dB loss vs 0.5 dB target — root cause is placeholder physics, not algorithm design
- **`implementation-strategy.md`**: Multi-phase roadmap: Phase 1 (1310/1550 nm SOI), Phase 2 (1064 nm SiN), Phase 3 (980 nm evaluation)
- **`final.md`**: Summary of what works vs what's still needed

---

## Common Tasks for AI Assistants

### Adding a new wavelength configuration
Modify `optimizer.py:__init__()` wavelength list. Ensure the new range is within Tidy3D's supported simulation bandwidth for the SOI platform. Update `examples/advanced_multi_wavelength.py` to demonstrate.

### Modifying manufacturing constraints
Change defaults in `ManufacturingConstraints.__init__()`. The filter radius auto-computes as `min_feature_size / grid_size` — do not set these independently without updating both.

### Adding a new initialization strategy
Add a `@staticmethod` to `InitializationStrategies` in `utils.py`. Return an `np.ndarray` of shape `(nx, ny)` with values in `[0.2, 0.8]` (avoid extreme values near 0 or 1 for initialization).

### Adding tests
Add to `tests/test_optimizer.py` using pytest conventions. Keep tests fast (no simulation calls). Mock Tidy3D calls with `unittest.mock.patch` when testing optimizer logic.

### Connecting real Tidy3D
See "Key Integration Points" section above. When implementing, also update `examples/basic.py` to set `td.config.logging_level = "ERROR"` to suppress verbose Tidy3D output in examples.

---

## Branch Strategy

- `main`: Stable, documented code
- `claude/add-claude-documentation-6QsL2`: Active development branch for AI-driven additions

All AI-authored changes should go to the designated development branch and be pushed via `git push -u origin <branch-name>`.

---

## Out-of-Scope

Do not modify:
- Physical constants (`eps_Si = 3.45**2`, `eps_SiO2 = 1.45**2`) without explicit user request — these are material properties
- The symmetry convention (4-fold) without validating that the adjoint gradient is also symmetric
- The SIMP penalty `p=3` — this value is a well-established choice for optical topology optimization

Do not add:
- JAX JIT compilation until real Tidy3D integration is working (premature optimization)
- New dependencies without updating `requirements.txt`
- Jupyter notebooks to version control (they are gitignored by convention)
