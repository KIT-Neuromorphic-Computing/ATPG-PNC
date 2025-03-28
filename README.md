# Automatic Test Pattern Generation for Printed Neuromorphic Circuits



This repository contains the source code, datasets, and experimental scripts associated with our paper:

**Automatic Test Pattern Generation for Printed Neuromorphic Circuits**\
*Tara Gheshlaghi, Priyanjana Pal, Alexander Studt, Michael Hefenbrock, Michael Beigl, Mehdi B. Tahoori*\
Karlsruhe Institute of Technology; RevoAI GmbH

*Accepted at ETS 2025*

---

## Overview

Printed Electronics (PE) promise a new era of low-cost, flexible, and energy-efficient hardware. In this work, we present a novel Automatic Test Pattern Generation (ATPG) framework designed specifically for Printed Analog Neuromorphic Circuits (pNCs). Our method combines:

- **Fault Analysis and Abstraction:** Clustering of faults with similar transfer functions and removal of untestable faults to significantly reduce the fault space.
- **Gradient-Based Optimization:** An efficient test pattern generation approach that maximizes output discrepancies between fault-free and faulty circuits.

Using our framework, experiments on nine pNC models demonstrated fault coverage exceeding 90% while significantly reducing the number of required test vectors compared to random pattern generation.

---

## Repository Structure

```
├── data/                   # Datasets and SPICE simulation data used in experiments
├── models/                 # Pre-trained pNC models and simulation configurations
├── notebooks/              # Jupyter notebooks for result visualization and interactive demos
├── scripts/                # Experiment and training scripts (e.g., run_experiment.py)
├── src/                    # Source code for fault modeling, abstraction, and gradient-based ATPG
├── configs/                # Configuration files for experiments
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # License information
```

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ATPG_pNC.git
   cd ATPG_pNC
   ```

2. **(Optional) Create a Virtual Environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Experiments

To replicate the experiments described in the paper—including fault injection, abstraction, and test pattern generation—run the following command:

```bash
python scripts/run_experiment.py --config configs/experiment_config.yaml
```

You can adjust the parameters in the configuration file to modify experimental settings (e.g., learning rate, number of epochs, or fault simulation parameters).

### Interactive Notebooks

For an interactive demonstration and detailed result visualization, open the Jupyter notebook:

```bash
jupyter notebook notebooks/ATPG_Demo.ipynb
```

This notebook provides step-by-step explanations of the fault modeling, abstraction, and gradient-based test pattern generation process.

---

## Code Overview

- **Fault Modeling:** Implements SPICE-based fault injection and models various defect types (open circuits, shorts, etc.) as detailed in the paper.
- **Fault Abstraction:** Uses clustering to group faults with similar transfer functions and removes untestable faults.
- **Test Pattern Generation:** Employs a gradient-based optimization routine (using PyTorch) to generate input patterns that maximize the discrepancy between the outputs of fault-free and faulty circuits.

---

## Citation

If you find our work useful, please cite our paper:

> T. Gheshlaghi, P. Pal, A. Studt, M. Hefenbrock, M. Beigl, and M. B. Tahoori,\
> "Automatic Test Pattern Generation for Printed Neuromorphic Circuits," ETS 2025.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

