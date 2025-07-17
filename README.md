# Airbnb Price Prediction with SIREN on Hong Kong Listings

This project uses **Airbnb public listing data** in Hong Kong to explore how well a neural network can predict apartment prices based solely on **geographic location** (latitude and longitude).

We experimented with **SIREN (Sinusoidal Representation Networks)** to model spatial patterns in the data. Although SIREN has shown promise in capturing high-frequency signals, in this case, it **did not perform well** for this task — the **price heatmap overlayed on a Hong Kong map** revealed weak generalization and unreliable predictions.

## 🔍 Project Goals

- Use publicly available Airbnb data
- Normalize and preprocess prices and coordinates
- Train a SIREN model to predict price from location
- Visualize model output using:
  - Kernel density estimates
  - A predicted price **heatmap** over a real-world **Hong Kong map**
  - Overlay of original training coordinates to compare against predictions

## 🧠 Model

- Architecture: **SIREN**
  - Input: 2D coordinates (latitude & longitude)
  - Output: Predicted normalized price
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Learning rate scheduler included

## 📊 Results

- The model was trained using a standard split of training, validation, and test sets.
- Despite successful training (loss convergence), the **predicted heatmap** showed significant errors in price predictions across the territory.
- This suggests that **location alone is not sufficient** to predict Airbnb prices, or that SIREN may not be well-suited for this type of regression problem on sparse geospatial data.

## 📁 Folder Structure
ml25-fresh/
├── data/ # Automatically downloaded Airbnb dataset
├── download_files.py # Dataset URL and extract logic
├── preprocess_data.py # Data cleaning, normalization
├── siren.py # SIREN model definition
├── main.py # Training script and heatmap generation
├── outputs/
│ └── hongkong_price_heatmap.html # Interactive HTML map
├── requirements.txt
├── notebooks/
│ └──main.ipynb  # Visualizations
├──1ml.png # Example output
└── README.md # This file


## 📦 Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt

## 👥 Authors: Team ML25 — Machine Learning Course, 2025, by Benjamin Eckhardt and Alina Amanbayeva



