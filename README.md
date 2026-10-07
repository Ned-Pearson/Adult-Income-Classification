# Adult Income Classification

Binary classification on the UCI Adult census dataset, predicting whether a person's
annual income exceeds $50,000. Uses the dataset's canonical train/test split.

Test ROC-AUC 0.9273, 95% CI [0.9230, 0.9314].

## Contents

| File | Description |
|---|---|
| `adult_income_classification.ipynb` | Full analysis with outputs |
| `report.md` | Written report |
| `report.pdf` | Same report as a PDF |
| `figures/` | Generated figures |
| `requirements.txt` | Dependencies |

## Running

```bash
pip install -r requirements.txt
jupyter notebook adult_income_classification.ipynb
```

Both data files are downloaded by the notebook at runtime, so no manual setup is needed.
A full run takes a few minutes on CPU, with the grid search being the slowest step.

## Data

Becker, B. & Kohavi, R. (1996). *Adult.* UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

32,561 training and 16,281 test records, 14 features, CC BY 4.0. This is 1994 census data
used here as a methodology benchmark, not as a basis for decisions about people.
