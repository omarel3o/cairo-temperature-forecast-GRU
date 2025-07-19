# Cairo Temperature Forecast Using GRU

Forecasting hourly temperatures in Cairo using a deep learning GRU model trained on three months of historical weather data.

---

## 📌 Project Overview

This project predicts hourly temperatures for the period **July 2 to July 9, 2025** using:

- A **stacked GRU architecture** trained on hourly temperature data from April 1 to July 1, 2025.
- A **multi-step forecasting approach** (24-hour forecasts) based on the previous 72 hours.
- Real-world evaluation by comparing predictions with actual measurements.

---

## 📂 Files Included

- `gru_forecast.ipynb` or `gru_forecast.py`: Main code used for training, forecasting, and evaluation.
- `requirements.txt`: Required Python packages.
- `README.md`: Project description and structure.
- `images/temp_comparison_plot.png`: Forecast vs. actual comparison plot.

---

## 🧠 Model Details

- **Input window**: 72 hours
- **Forecast horizon**: 24 hours
- **Architecture**:
  - `GRU(128)` with return sequences
  - `Dropout(0.3)` to reduce overfitting
  - `GRU(64)`
  - `Dense(24)` for multi-hour prediction
- **Optimizer**: Adam
- **Loss function**: Mean Squared Error
- **Training strategy**: EarlyStopping to avoid over-training

---

## 📊 Evaluation Results

After training and comparing against actual hourly temperatures, the model achieved:

- **MAE (Mean Absolute Error)**: `2.97°C`
- **RMSE (Root Mean Squared Error)**: `3.80°C`

![Temperature Comparison Plot](images/temp_comparison_plot.png)

---

## 🗃️ Data Source

Hourly historical and actual temperature data were downloaded from [Meteoblue](https://www.meteoblue.com) for Cairo, Egypt.

---

## 🛠 How to Use

1. Upload the 3-month training CSV file (April–July 1).
2. Upload the actual hourly temperature CSV (July 2–9).
3. Run the notebook or script in Google Colab or locally.
4. Compare predictions and analyze performance.

---

## ⚙️ Requirements

```txt
tensorflow
pandas
scikit-learn
matplotlib
