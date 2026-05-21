# Prism-CRISPR

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20278469-blue)](https://doi.org/10.5281/zenodo.20278469)

Prism-CRISPR is a deep learning framework for CRISPR sgRNA prediction. It contains two model components:

- **Prism-on**: predicts on-target cleavage efficiency for Cas9 sgRNAs.
- **Prism-be**: predicts base editing outcome distributions for base editors, including ABEs and CBEs.

This repository is intended to provide the saved model files, model architecture definitions, and supporting scripts for prediction and reference. 

## Installation

Please create a clean conda environment and install each dependency explicitly as shown below.

```bash
# Create and activate a new environment
conda create -n prism-crispr python=3.12.11
conda activate prism-crispr

# Install PyTorch
pip install torch==2.8.0

# Install numerical computing libraries
pip install numpy==2.4.1
pip install scipy==1.16.3

# Install data processing library
pip install pandas==2.3.3

# Install machine learning utilities
pip install scikit-learn==1.8.0

# Install plotting library
pip install matplotlib==3.10.8

```






