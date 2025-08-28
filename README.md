# MLP Learning Rate Tuning - Nonlinear Regression (2 inputs -> 1 target)

## Overview
This project shows how to pick a good learning rate for a single hidden layer MLP on a small nonlinear problem with two input variables and one continuous target. It includes a simple sweep over candidate learning rates, clear selection rules, and a final model with reported performance.

## Problem statement
Predict a continuous target from two numeric inputs. The relationship is nonlinear, so a multilayer perceptron (MLP) is used. The key design choice explored here is the learning rate, which controls the size of weight updates during training.

## Data at a glance
- Rows: 500
- Features: 2 numeric inputs
- Target: 1 numeric output
- Split: 70 percent train, 30 percent test (fixed random seed)

## Method in brief
1. Model
   - scikit-learn `MLPRegressor` with one hidden layer of 100 units, solver='adam', learning_rate='constant', max_iter=1000, random_state=42.
2. Learning rate sweep
   - Try `learning_rate_init` values on a small grid (for example 0.30 to 0.49 in steps of 0.01).
   - For each value, train on the train split and compute R2 and MSE on both train and test.
   - Use plots of R2 vs learning rate and MSE vs learning rate to check underfitting vs instability.
3. Selection rule
   - Choose the learning rate with the lowest test MSE (and high test R2) while keeping the train vs test gap small.
4. Final model
   - Refit the MLP at the chosen learning rate on the train split and report performance on the held-out test set.

## Results
- Chosen learning rate: 0.34
- Train MSE: 810.810
- Test MSE: 838.035
- Generalisation gap (test minus train MSE): 27.225
- Train R2: 0.7731
- Test R2: 0.7731

These results show a good fit to the nonlinear signal with a modest gap between train and test errors. Lower learning rates trained slowly and underfit. Higher values led to unstable training or worse test error. The sweep showed a clear sweet spot near 0.34.

## What the model does
A single hidden layer MLP learns nonlinear interactions between the two inputs and the target. With the tuned learning rate, training converges smoothly to a solution that generalises well to unseen data.

## Business view
- A well-chosen learning rate improves accuracy and stability while reducing training time and compute cost.
- The simple sweep and selection rule are easy to repeat on new data, and can be added to a lightweight model governance checklist.
- Better generalisation means fewer surprises when the model is moved from experimentation to a real setting.

## Repository layout
.
- data/                  place your CSV file here
- notebooks/             mlp_learning_rate_tuning.ipynb
- reports/               mlp_learning_rate_tuning - Jupyter Notebook.pdf
- README.md

## How to reproduce
1) Install dependencies
```
pip install pandas numpy scikit-learn matplotlib
```
2) Put your CSV in `data/` and load it like:
```
import pandas as pd
df = pd.read_csv("data/your_file.csv")  # replace with your filename
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
```
3) Open and run the notebook
```
jupyter notebook "notebooks/CW1 Task 2 Solution (230295100).ipynb"
```
The notebook runs the learning-rate sweep, prints R2 and MSE for each candidate, and plots the curves used for selection.

## Next steps
- Try other hidden layer sizes and activations and compare results.
- Add k-fold cross-validation for more robust estimates.
- Put a `StandardScaler` in a `Pipeline` to help optimiser behaviour.
- Add a simple baseline (for example linear regression) for context.
- Persist the trained model with `joblib` and add a small predict script.
- Track a random seed and pin package versions for reproducibility.

## License
MIT

Author: Mustafa Scentwala
