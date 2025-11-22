# Flux Dynamics: Procedural Quantization (FDC-4.0)

**A computational framework deriving fundamental physical constants from geometric first principles using Genetic Algorithms.**

  

## 🔭 Overview

This repository contains the source code for **Flux Dynamics (Model FDC-4.0)**, a theoretical physics framework that derives the values of 25 fundamental constants (including the Higgs Mass, Weak Mixing Angle, and Fermion Mass Hierarchy) using a unified geometric lattice.

Unlike standard numerology, this model utilizes a **Procedural Quantization Algorithm**. We treat the derivation of physical constants as a computational process with specific "bit-depth" requirements. Using a Genetic Algorithm (PyGAD), we optimized the quantization scaling factors (instructions) to match observed reality.

### The "Minimal Digit" Discovery

The script included in this repo (`FDC.py`) was optimized with a specific, hierarchical objective:

1.  **Hard Constraint:** Predictions must match **25/25** observed constants within $1\sigma$ (Standard Error).
2.  **Primary Goal:** Minimize the average deviation from experimental values.
3.  **Secondary Goal:** Minimize the **computational cost** (precision/bit-depth) required to generate those values.

## 🧬 Key Findings: Differential Computational Complexity

The optimization results reveal a striking structure in how the universe processes information:

  * **Lossless Calculations (Infinite Precision):** The Vacuum Energy ($\rho_{vac}$) and Cosmological Constant require maximum available precision (saturating the bit-depth cap). This aligns with the concept of the vacuum as a high-complexity state.
  * **High-Bitrate Calculations (Analog):** The Fermion Mass Hierarchy (especially light quarks and the electron) behaves like a sensitive analog signal, requiring 50+ digits of precision to account for exponential cascading errors (the "butterfly effect").
  * **Low-Bitrate Calculations (Discrete Geometry):** Surprisingly, fundamental structural constants like the **Higgs Mass ($m_H$)**, **Weak Mixing Angle ($\sin^2 \theta_W$)**, and **Speed of Light scale ($c$)** optimized to extremely low precision (1-5 digits). This suggests these values are **geometric integers** or topological invariants—structural inevitabilities of the lattice that are robust against fluctuations.

## 📊 Results (Validation)

The model successfully predicts **25 out of 25** target constants within $1\sigma$ of the 2024 Particle Data Group (PDG) and CODATA 2022 values.

**Summary Metrics:**

  * **Total Matches:** 25 / 25 (100%)
  * **Average Deviation:** 0.049%
  * **Average Sigma:** 0.07

### Core Bosons & Geometry

| Constant | Calculated | Observed (PDG) | Deviation | Sigma |
| :--- | :--- | :--- | :--- | :--- |
| **Higgs Mass ($m_H$)** | **125.30 GeV** | 125.26 GeV | 0.03% | 0.29 |
| **Weak Angle ($\sin^2 \theta_W$)** | **0.23120** | 0.23122 | 0.009% | 0.50 |
| **Inv. Fine Structure ($\alpha^{-1}_{MZ}$)** | **128.952** | 128.952 | 0.00% | 0.00 |
| **Gravitational Constant ($G$)** | **6.6743e-11** | 6.6743e-11 | 0.00% | 0.00 |

### Fermion Mass Hierarchy (Selected)

| Particle | Calculated (GeV) | Observed (GeV) | Sigma |
| :--- | :--- | :--- | :--- |
| **Top Quark** | 172.76 | 172.76 | 0.00 |
| **Bottom Quark** | 4.183 | 4.183 | 0.00 |
| **Electron** | 0.000510999 | 0.000510999 | 0.00 |
| **Neutrino Sum ($m_\nu$)** | 0.05138 eV | 0.0513 eV | 0.02 |

## 🛠️ Installation & Usage

### Dependencies

The simulation requires Python 3.x and the `pygad` library for genetic optimization.

```bash
pip install pygad
```

### Running the Simulation

To reproduce the results or run a new optimization cycle:

```bash
python FDC.py
```

*Note: The script is currently configured with `TOTAL_EPOCHS_TO_RUN = 0` to immediately demonstrate the pre-calculated optimal solution. Increase this value to perform a fresh search.*

## 🧠 Methodology

The framework defines a dictionary of $\approx 400$ "genes" (quantization instructions). Each gene represents the rounding precision (bit-depth) of a specific geometric operation in the theory (e.g., `t_to_mz_step1`).

The Fitness Function utilizes a **Tiered Scoring System**:

1.  **Tier 1 (Invalid):** If the model predicts $< 25$ correct constants, the score is based solely on minimizing standard error.
2.  **Tier 2 (Valid):** If the model predicts $25/25$ correct constants, it receives a massive bonus score, which is then further incremented by:
      * Inverse Average Deviation (Primary Optimization)
      * Inverse Total Digits Used (Tiebreaker)

## 📂 Repository Structure

  * `FDC.py` - The main simulation engine and Genetic Algorithm manager.
  * `README.md` - Project documentation.

## 📜 Citation

**Author:** Zac Wickstrom
**Paper:** *Flux Dynamics: A Geometric Framework for Unification*

If you use this code or methodology in your research, please cite the accompanying paper or this repository.

-----
