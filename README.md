# M.Tech-Project
# Efficient Training and Memory Optimization for Margin Propagation Networks

This repository contains the code and experiments from my M.Tech thesis at IISc Bangalore, under the guidance of Prof. Chiranjib Bhattacharyya and Prof. Chetan Singh Thakur.

## Overview

Margin Propagation (MP) is a multiplication-free neural framework using addition and thresholding, enabling energy-efficient inference on edge devices. This thesis introduces:
- Theoretical error bounds for MP approximation.
- Efficient weight porting from MLP to MP networks.
- A connection between MP and Simplex Projection for memory-efficient training.
- A perceptron-like interpretation of n-LIF spiking neurons.

## Structure

- `src/` – Core models and algorithm implementations.
- `experiments/` – Scripts for training, evaluation, and analysis.
- `notebooks/` – Jupyter notebook walkthroughs.
- `thesis/` – The full M.Tech thesis PDF.
- `results/` – Plots, tables, and benchmark comparisons.

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
