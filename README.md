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

## Usage

After installation, you can use the saved models (.pth) to perform prediction on your own sgRNA dataset.

## Example Usage (Prism-on)

### Example command

```bash
python predict.py WT-SpCas9 WT-SpCas9.csv WT-SpCas9_pred.csv
```

### Arguments

1. Model type:
   - WT-SpCas9 → WT-SpCas9_best_model.pth
   - eSpCas9 → eSpCas9_best_model.pth
   - SpCas9-HF1 → SpCas9-HF1_best_model.pth

2. Input CSV file:
   Example: `WT-SpCas9.csv`

3. Output CSV file:
   Example: `WT-SpCas9_pred.csv`

### Example Input CSV

```csv
sgrna,true_eff
GACGGTAACGGACGTAATCACGG,0.683407501589319
ACAGTGACCACGGATCAAGATGG,0.592688602486493
GCTAAGGAAACTCATCTCCGAGG,0.934380430877969
GTTACAGTTCGGTCCAACAGTGG,0.932465160191285
GGGCGACTCAGACAGCGCATCGG,0.761833316032281
```

### Example Output CSV

```csv
sgrna,true_eff,pred_eff
GACGGTAACGGACGTAATCACGG,0.683407501589319,0.6815759
ACAGTGACCACGGATCAAGATGG,0.592688602486493,0.6576845
GCTAAGGAAACTCATCTCCGAGG,0.934380430877969,0.86801565
GTTACAGTTCGGTCCAACAGTGG,0.932465160191285,0.89584124
GGGCGACTCAGACAGCGCATCGG,0.761833316032281,0.7382468
```

### Output Description

- `pred_eff`: predicted editing efficiency
- The script will also print:
  - Pearson correlation coefficient
  - Spearman correlation coefficient

These metrics evaluate agreement between predicted and true efficiencies.









