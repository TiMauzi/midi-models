# 🎵 MIDI-to-Year Prediction
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

This project features a machine learning pipeline that predicts the **composition year** of a musical work based solely on features extracted from MIDI files.

## 👀 Overview
The notebook demonstrates how to transform raw MIDI data into meaningful features to train a regression model. The workflow includes:

* **Data Acquisition**: Loading the [TiMauzi/imslp-midi-by-sa](https://huggingface.co/datasets/TiMauzi/imslp-midi-by-sa/) dataset from the Hugging Face Hub.
* **Feature Engineering**: Extracting the top 15 most common MIDI message types (e.g., `note_on`, `control_change`, `set_tempo`) and calculating their frequency per piece.
* **Model Training**: Implementing a `GradientBoostingRegressor` to handle the non-linear relationships in musical structure across eras.
* **Evaluation**: Testing the model's precision using Mean Absolute Error (MAE) and accuracy windows (e.g., how often the model estimates within the decade or century of the actual date).

## 📊 Results Summary
The simple regression model already shows a strong ability to identify the general era of a piece, even without access to the actual audio or sheet music.

| Metric | Result |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | ~42.17 years |
| **Root Mean Squared Error (RMSE)** | ~58.74 years |
| **±25-Year Accuracy** | 34.4% |
| **±50-Year Accuracy** | 59.1% |

While predicting a specific year is challenging, the model correctly places nearly **60% of pieces within a one-century window**, proving that MIDI message distributions can be crucial indicators for musical analysis.
