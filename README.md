# Credit Card Fraud Detection
## Overview

This project is about detecting fraudulent card transactions in a heavily skewed dataset. The core idea is showing why accuracy is the wrong way to measure success when fraud is only 0.17% of this dataset, a figure that is low because fraud in the real world is genuinely this rare.

Card fraud is a needle-in-a-haystack problem: the vast majority of transactions are legitimate, so a model that flags nothing as fraud is still accurate 99.83% of the time and catches none of it. This project treats the problem the way a bank's analytics team has to: the metrics that matter are recall (how much of the fraud we catch, since low recall means fraud slips through) and precision (when we raise an alert, are we right, since low precision means freezing innocent customers' cards), not headline accuracy.

The project takes an anonymised dataset of real European card transactions, builds the data-cleaning and modelling steps, then compares a simple baseline model against models that have been adjusted to take the rare fraud cases seriously.

## Dataset

This is the Credit Card Fraud Detection dataset from the Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB). The page notes it was collected during a research collaboration between Worldline (a payment company) and that group (MLG-ULB). The dataset is very widely used with over a million downloads and the link is as follows:

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, MLG-ULB).

- 284,807 transactions made by European cardholders over two days in September 2013, 492 of them fraudulent (0.172%).
- Features `V1`–`V28` are the output of a PCA transformation (anonymised for confidentiality), plus `Time` (number of seconds between each transaction and the very first transaction in the dataset), `Amount` (value of the transaction), and the target `Class` (1 = fraud, 0 = legitimate).

PCA (principal component analysis) was used here for confidentiality, because the real fields are sensitive banking data that could not be published. They were transformed into anonymised components a model can still learn from, since the patterns are still present.

- Download `creditcard.csv` and place it in a `data/` folder in the project root. The file is not committed to the repo (see `.gitignore`).

## Approach

1. **Explore the imbalance.** Print the class split and the useless always legit baseline, comparing fraud versus valid transactions. State ratio to show why accuracy is misleading as a measure. 
2. **Preprocess.** Two jobs: Scaling | Stratified train/test split. Scaling: PCA components are on a similar small scale, so `Amount` and `Time` need scaling as they are on a much bigger number range. Stratified train/test split: train/test split done 80/20 respectively so the model learns from the training and has the testing portion of data to judge it on. Stratified so both sets contain the same tiny fraud ratio instead of a risk of randomly splitting. 
3. **Handle imbalance two ways.** Class weighting | SMOTE oversampling. Class weighting: make each fraud case count more heavily, penalising them harder for missing a fraud over misjudging a legit transaction. SMOTE oversampling: only done in training, not test set. Creating synthetic fraud examples by blending real ones so data looks more balanced. Approaching imbalance in both ways to see the trade-offs instead of assuming which is best.
4. **Model.** A logistic regression benchmark so any improvement is proven, not assumed. Compare the LR benchmark against a random forest. SVM not chosen as tends to scale poorly with large datasets.
5. **Evaluate on what matters.** Judge the models on metrics suited to rare events, not accuracy:
- Precision: TP / (TP + FP). Of all flagged fraud, how much truly was.
- Recall: TP / (TP + FN). Of all real fraud, how much was actually caught.
- F1: a single score balancing precision and recall.
- ROC-AUC: how the catch rate trades off against the false-alarm rate as the threshold changes.
- PR-AUC: precision vs recall across the thresholds.
- Plus a confusion matrix and a PR curve as visual outputs.

PR-AUC is the headline metric. Both its numbers focus on the rare fraud class, so it isn't skewed to dilution by the huge legit class the way ROC-AUC is. The dataset's own authors recommend it for exactly this reason.

## Results

These will be filled in after the first run. The point of the table is to show the accuracy-versus-recall story clearly.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|-----|---------|--------|
| Logistic Regression (balanced) | | | | | |
| Random Forest (balanced) | | | | | |
| Random Forest + SMOTE | | | | | |

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# place creditcard.csv in ./data/
python fraud_detection.py
```

Outputs a metrics summary to the console and saves a precision-recall curve to `pr_curve.png`.

## Repo structure

```
.
├── data/                 # creditcard.csv lives here (gitignored)
├── fraud_detection.py    # full pipeline
├── requirements.txt
├── pr_curve.png          # generated on run
└── README.md
```

## Key takeaways

- On this dataset the model scores 99.83% when assuming the transactions are always legit and catches zero fraud. This is the main point portrayed throughout the project as a whole.
- Precision and recall trade off: catch more fraud and you raise more false alarms. Where you set that balance is a business decision (annoyed customers), not a modelling one.
- Precision-recall AUC, not ROC-AUC or accuracy, is the metric that reflects real performance due to it being built from purely fraud related numbers.

## Possible extensions

- Threshold tuning to hit a target precision or recall, framed as a cost trade-off (a missed fraud versus an annoyed customer).
- Feature importance / SHAP values for explainability. SHAP is a technique that shows which features pushed the model towards fraud for a specific transaction. In this project the features are anonymised PCA components, so SHAP could tell me V14 mattered most, but since V14 is an anonymised component I can't say what it represents in the real world. 
- A small Streamlit front end that turns the script into a simple web app, scoring a single transaction live. A bridge into the dashboard project. 
