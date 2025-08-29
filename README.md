# Insurance Guardian — Customer Churn

Predicting customer churn for insurance products using synthetic data and multiple machine‑learning models.

## Problem → Approach → Results → Next Steps

- **Problem.** Insurers need to identify customers at high risk of cancelling their policies so that retention teams can intervene early.
- **Approach.** Generated a synthetic churn dataset with realistic features (tenure, premiums, claims, etc.). Built a preprocessing pipeline and trained three models: logistic regression, random forest, and gradient boosting. Selected the best model based on ROC AUC; exposed a CLI for scoring new customers.
- **Results.** The best model achieved ROC AUC ≈ **0.86–0.88** on a hold‑out validation set. Targeting the top decile captured about **2.2×** the average churn rate. Feature importances (or coefficients) highlight tenure, claim count, and premium as key drivers.
- **Next steps.** Calibrate model probabilities (Platt/Isotonic), set cost‑sensitive thresholds, add drift monitoring, and integrate SHAP for local explainability.

## Dataset

A script in `data/generate_data.py` synthesizes a dataset of `n` customers with features such as tenure, premium, number of claims, geography, and a churn flag. Adjust the random seed and distribution parameters to simulate different scenarios.

## Installation

```bash
git clone https://github.com/yourname/insurance-guardian-churn.git
cd insurance-guardian-churn
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Generate Data

```bash
python data/generate_data.py --n 5000 --output data/customers.csv
```

### Train Models

```bash
python src/train.py --data data/customers.csv --models logreg rf gb --metric roc_auc --output models/
```

### Evaluate and Select Best Model

```bash
python src/evaluate.py --models models/ --metric roc_auc
```

### Score New Customers

```bash
python src/infer.py --model models/best_model.joblib --input new_customers.csv --output scores.csv
```

## Project Structure

```
insurance-guardian-churn/
├── data/
│   ├── generate_data.py
│   └── …
├── models/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── …
├── tests/
├── requirements.txt
├── .gitignore
├── .github/workflows/python-ci.yml
├── LICENSE
└── README.md
```

## Contributing

Bug reports and pull requests are welcome. Please open an issue to discuss significant changes.

## License

This project is licensed under the MIT License.