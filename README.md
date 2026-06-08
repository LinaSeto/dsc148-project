# DSC 148 Project: ICU In-Hospital Mortality Prediction

This project predicts in-hospital mortality for ICU patients. The project compares a logistic regression baseline against a sigmoid-calibrated CatBoost model. The workflow includes data cleaning, exploratory data analysis, and model building. The project further analyzes the models with a threshold analysis, APACHE sensitivity analysis, and a SHAP feature attribution. The final component is a interactive demo using a simplified model.

## Dataset

The dataset used is the Kaggle WiDS Datathon 2020 GOSSIS dataset. It can be found at https://www.kaggle.com/c/widsdatathon2020/data.

There are 91,713 ICU encounters (rows) and 186 features (columns).

The data is not included in this repository. To retrieve it, download the `training_v2.csv` file from the link provided. It can then be called using `pd.read_csv('training_v2.csv')` in a notebook.

## Results

This is the test-set performance of the full model with a seed 42.

| Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| Logistic Regression + sigmoid (baseline) | 0.8894 | 0.539 | 0.0556 | 0.008 |
| CatBoost + sigmoid calibration (proposed) | 0.9013 | 0.575 | 0.0546 | 0.016 |

## Setup

Use Python 3.9 or later and have the following installed:

```bash
pip install pandas numpy scikit-learn catboost matplotlib
```

## Replication

All steps are in the notebook `dsc148_project_notebook.ipynb` and should be run from top to bottom in order. Seed 42 is fixed throughout the notebook and using it should reproduce reported metrics. 

| Step | Description |
|---|---|
| 1. Load data | `pd.read_csv("training_v2.csv")` |
| 2. EDA | Explore outcome balance, missingness, vitals by outcome, subgroup mortality, missingness-as-signal, univariate AUC, redundancy, age & comorbidity |
| 3. Clean data | `clean_deterministic(df)` — dtype coercion, physiological bounds, max/min consistency checks |
| 4. Split | 60 / 20 / 20 (train / calibration / test), seed 42 |
| 5. Feature sets | `build_feature_frame(df, setting)` — settings A–D for the APACHE sensitivity analysis |
| 6. Train | Logistic Regression baseline + proposed CatBoost, each within a leakage-safe pipeline |
| 7. Calibrate | Sigmoid scaling fit on the calibration split |
| 8. Evaluate | ROC/PR curves, Brier/ECE, threshold analysis, calibration curve, SHAP |
| 9. Demo export | Writes `docs/index.html`, a self-contained browser demo |

## Interactive Demo

Open `docs/index.html` in a browser to view the interactive demo built with a simpler model than the full model evaluated here. 

It is also accessable through a shareable link through enabling GitHub Pages in repository settings. 

## Usage Disclamer

This project is for education purposes only. The models are not meant to inform clinical decisions on patients. 
