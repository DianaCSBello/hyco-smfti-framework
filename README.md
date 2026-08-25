# hyco-smfti-framework

Hybrid computational framework for Spatially Modulated Floquet Topological Insulators (SM-FTIs) in waveguide arrays with sublattice-dependent curvature.

## Overview

This repository contains the code and data used in the final project report *"Hybrid Computational Framework for Spatially Modulated Floquet Topological Insulators"*. The pipeline couples high-fidelity continuum simulations (FreeFem++) with a tight-binding model and a machine-learning residual corrector (MLP) to provide fast and accurate predictions of edge-state observables.

## Repository structure

```text
├── FreeFem/
│   ├── hyco_pml_laude_final.edp      # Solver for reference case
│   ├── hyco_pml_laude_param.edp      # Template for parameter sweep
│   └── ...
├── Python/
│   ├── run_sweep.py                  # Parameter sweep launcher
│   ├── hybrid_baseline.py            # TB + MLP (baseline)
│   ├── hybrid_advanced.py            # TB + per-metric MLP (advanced)
│   ├── analyze_general.py            # General training across sweep
│   ├── predict.py                    # Prediction for new parameters
│   ├── extract_figures.py            # Figures and animation
│   └── ...
├── data/
│   ├── overlap_data_pml_laude_final.dat
│   ├── overlap_data_pml_laude_second.dat
│   ├── overlap_data_A1_*.dat         # Parameter sweep outputs
│   └── ...
├── figures/
│   ├── hybrid_mlp_advanced_results.pdf
│   ├── hibrido_comparacion_final.pdf
│   ├── evolucion_completa_metrics.pdf
│   ├── evolucion_completa_campo.gif
│   └── ...
└── README.md
```

## Requirements

- **FreeFem++** (version 4.x or later)
- **Python 3.8+** with packages: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `joblib`

## Usage

### 1. Run the reference continuum simulation

```bash
cd FreeFem
FreeFem++ hyco_pml_laude_final.edp
```

This produces `overlap_data_pml_laude_final.dat` and field files.

### 2. Generate parameter sweep data

```bash
cd Python
python run_sweep.py
```

The script uses `hyco_pml_laude_param.edp` and launches simulations in parallel.

### 3. Train the general hybrid model

```bash
python analyze_general.py
```

It loads all `overlap_data_A1_*.dat`, computes FEM and TB metrics, trains the MLP residual corrector, and saves the model.

### 4. Predict for new parameters

Edit the parameters in `predict.py` and run:

```bash
python predict.py
```

### 5. Generate figures and animation

```bash
python extract_figures.py
```

## Results

- Beat calibration: τ₁ = 0.5649, τ₂ = 0.8644 (ratio 1.53)
- Advanced hybrid RMSE (reference case): 0.8778 (sum)
- General hybrid test RMSE: 0.8313 (sum)

## Reference

The associated journal article and non-technical post are included in the final project report.

## License

This project is for research purposes. Please cite the report if you use the code or data.
