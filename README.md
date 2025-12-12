# ft_linear_regression

**Predict used-car prices from mileage using a simple linear regression implemented from scratch in Python.**

---

## 1. Project Overview

This project implements a **linear regression model** to estimate car prices based on mileage. The model is trained using **gradient descent** and supports:

- Data normalization for stable training  
- Gradient descent with explicit update rules  
- Denormalization of learned parameters for real-world predictions  
- Cost tracking and convergence visualization  
- Precision evaluation using **R² score**  
- Saving/loading model parameters for later predictions  
- Predicting the price of a car given its mileage  

---

## 2. Dataset

The project expects a CSV file named `data.csv` with two columns:

| Column Name | Description                     |
|------------|---------------------------------|
| `km`      | Car mileage in kilometers       |
| `price`   | Selling price in euros          |

Example:

| km      | price  |
|---------|--------|
| 50000   | 12000  |
| 120000  | 7000   |
| 90000   | 8500   |

---

## 3. Installation

### Requirements

- Python 3.8+  
- Libraries: `numpy`, `pandas`, `matplotlib`  

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Training the Model

Run the main training script:

```bash
python train.py
```

You will see a menu:

```
=== Training Options ===
1. Train the model with raw data
2. Visualize the regression results
3. Visualize the cost function over iterations
4. Quit
```

- **Option 1** → trains the model and prints results including **denormalized parameters**, **final cost**, and **R² precision**.  
- **Option 2** → plots the data and regression line.  
- **Option 3** → plots cost vs. iterations.  

---

## 5. Model Parameters

The model saves parameters into `values.json`:

```json
{
  "Theta0": 8481.17,
  "Theta1": -0.02127
}
```

- **Theta0** → intercept (estimated price for 0 km)  
- **Theta1** → slope (price decrease per km)

---

## 6. Predicting Prices

A separate script `predic.py` allows the user to input mileage and get an estimated price:

```bash
python predic.py
```

Example:

```
Enter car mileage (km): 89000
Estimated price: 8481.16 €
```

---

## 7. Model Precision

The project calculates **R² score** as a measure of accuracy:

\[
R^2 = 1 - \frac{\sum(y_\text{true} - y_\text{pred})^2}{\sum(y_\text{true} - \bar{y})^2}
\]

- R² = 1 → Perfect prediction  
- R² = 0 → Model predicts just the mean  
- R² < 0 → Model is worse than mean  

Example result from training:

```
Model Precision (R² score) = 0.7329
```

This means the model explains **~73% of the variation in price**, which is reasonable for real-world used-car data.

---

## 8. Visualization

- **Data vs. Regression Line**: shows real data points and the predicted line.  
- **Cost over Iterations**: shows the decrease of the cost function during training.

Plots are saved as:

- `plot.png` → regression line and scatter plot  
- `cost_plot.png` → cost function over iterations  

---

## 9. Project Structure

```
ft_linear_regression/
├── data.csv              # Dataset (mileage, price)
├── train.py              # Main training script
├── predic.py             # Predict price for a given mileage
├── values.json           # Saved model parameters (after training)
├── plot.png              # Regression plot (generated)
├── cost_plot.png         # Cost plot (generated)
└── README.md             # Project documentation
```

---

## 10. How It Works

1. **Data Normalization** → Scales `km` and `price` to [0,1] for numerical stability.  
2. **Gradient Descent** → Iteratively updates θ₀ and θ₁ to minimize cost.  
3. **Denormalization** → Converts θ₀ and θ₁ back to real price units.  
4. **Precision Evaluation** → Computes R² to measure prediction accuracy.  
5. **Prediction** → Given a mileage, the model returns the estimated price using saved parameters.  

---

## 11. Example Output

```
==================== TRAINING SUMMARY ====================

Normalized Theta0 : 0.9362
Normalized Theta1 : -0.9954

Real Theta0       : 8481.17 €
Real Theta1       : -0.02127 €/km

Final Training Cost (MSE) : 0.01035
Model Precision (R² score) : 0.7329

==========================================================
```

---

## 12. Notes

- Your R² may vary slightly depending on dataset and initialization.  
- Larger datasets may require adjusting the learning rate or number of iterations.  
- Normalization is essential for correct gradient descent convergence.  

---

## 13. Author

**Narcisse Obadiah** – ft_linear_regression project, 42 Heilbronn.

