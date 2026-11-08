# Bank Marketing Classification Model Lab

**BITS Pilani WILP | M.Tech (DSE) | Machine Learning | Assignment 2**

## a. Problem statement

Predict whether a client of a Portuguese banking institution subscribes to a term deposit after direct-marketing activity. This is a binary classification problem. The target is `y` (`yes` / `no`).

The work implements Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine. All models use a common stratified 80:20 train/test split and consistent preprocessing. Performance is evaluated using Accuracy, ROC-AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC).

**Leakage control:** `duration` is excluded. UCI explains that call duration is only known after the call ends and should be discarded for a realistic predictive model.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, Bank Marketing (dataset ID 222).
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Selected version:** `bank-full.csv` (legacy version).
- **Dataset size:** 45,211 instances, 16 original input features and one binary target (`y`).
- **Feature types:** categorical and integer/numeric.
- **Target:** whether the client subscribed to a term deposit (`yes` or `no`).
- **Predictors used in this project:** 15 (16 original predictors minus `duration` to avoid target leakage).
- **Class imbalance:** the `yes` class is the minority class; therefore accuracy is interpreted together with AUC, precision, recall, F1 and MCC.

Official source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## c. GitHub Repository Link

**Add your public GitHub repository URL here after creating and pushing the repository:**

`[PASTE YOUR GITHUB REPOSITORY URL]`

## d. Models used

1. Logistic Regression — linear probabilistic classifier; class weighting is enabled.
2. Decision Tree Classifier — non-linear tree model with controlled depth and minimum leaf size.
3. K-Nearest Neighbors — distance-weighted 7-neighbour classifier.
4. Gaussian Naive Bayes — probabilistic classifier using a Gaussian likelihood assumption.
5. Support Vector Machine — LinearSVC with class weighting. Its decision scores are used to calculate ROC-AUC.

### Preprocessing and experiment protocol

- Split: 80% training / 20% testing, stratified by target.
- Random seed: 42.
- Numeric columns: median imputation and standard scaling.
- Categorical columns: most-frequent imputation and one-hot encoding with unknown-category handling.
- Preprocessing is fitted only on the training split using scikit-learn Pipelines.
- `duration` is excluded from both training and test inputs.
- Class weighting is used for Logistic Regression, Decision Tree and SVM to reduce the impact of class imbalance.
- ROC-AUC uses predicted probabilities where available and decision scores for LinearSVC.

### Model comparison table

Run `python model/train_models.py` to generate the real values in `results/model_comparison.csv`. Copy those measured values into this table before final submission. Do not report placeholder or assumed metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| Decision Tree | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| KNN | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| Naive Bayes | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |
| Support Vector Machine | Pending run | Pending run | Pending run | Pending run | Pending run | Pending run |

### Observations on model performance

Complete this section from the generated comparison table and confusion matrices after running the experiment. Use the same held-out test set for every model.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Add evidence-based observation about precision/recall trade-off, AUC and MCC. |
| Decision Tree | Comment on non-linear patterns, generalization and minority-class errors. |
| KNN | Comment on distance-based performance after scaling and minority-class detection. |
| Naive Bayes | Comment on its probabilistic baseline and feature-independence assumption. |
| Support Vector Machine | Comment on margin-based separation, AUC and class-weighted predictions. |
| **Overall winner for this dataset** | Select only after comparing the measured metrics. State which metric(s) informed the choice. |

**Winner-selection guidance:** With an imbalanced target, do not select a model using accuracy alone. Consider ROC-AUC, F1 and MCC together, and explain the precision/recall trade-off relevant to the use case.

## Streamlit application features

- CSV upload for a labelled test dataset.
- Model-selection dropdown.
- Accuracy, ROC-AUC, Precision, Recall, F1 and MCC display.
- Confusion matrix.
- Classification report.
- Preview of actual versus predicted labels.

## Project structure

```text
Bank_Marketing_ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv                 # Generated with held-out test rows after training
├── data/
│   ├── download_dataset.py
│   └── bank-full.csv              # Downloaded dataset; do not commit if not permitted
├── model/
│   ├── train_models.py
│   └── saved_models/              # Generated .joblib model pipelines
├── results/
│   ├── model_comparison.csv
│   ├── evaluation_details.json
│   └── run_metadata.json
├── notebooks/
│   └── Bank_Marketing_Classification.ipynb
└── screenshots/
    └── bits_virtual_lab.png       # Add your own BITS Virtual Lab screenshot
```

## Installation and execution

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python data/download_dataset.py
python model/train_models.py
streamlit run app.py
```

After training, the script creates:
- `model/saved_models/*.joblib`
- `test_data.csv` (the stratified held-out test rows, including target `y`)
- `results/model_comparison.csv`
- `results/evaluation_details.json`
- `results/run_metadata.json`

Open the local Streamlit URL printed in the terminal. Upload `test_data.csv` or another labelled CSV to evaluate the selected model.

## GitHub and Streamlit Community Cloud deployment

1. Create a new public GitHub repository for this project.
2. Add the source files, `requirements.txt`, completed README, generated model files and labelled `test_data.csv`.
3. Do not commit virtual environments, cache folders or personal credentials.
4. Commit and push the project to the `main` branch.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Select **Create app**, choose the repository, branch `main`, and entrypoint `app.py`.
7. Deploy and wait for the app health check to complete.
8. Open the public `*.streamlit.app` URL in a private/incognito window to verify it.
9. Add the actual GitHub URL and deployed Streamlit URL to the final PDF.

Community Cloud dependency files should be in the repository root or beside the app entrypoint. See https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## BITS Virtual Lab evidence

Run the notebook or Python scripts in BITS Virtual Lab. Capture one genuine screenshot showing the project execution (for example, the terminal/notebook with successful model evaluation). Save it as `screenshots/bits_virtual_lab.png` and insert it into the final PDF. Do not substitute a local-machine screenshot for the required BITS Virtual Lab proof.

## Academic integrity and reproducibility

This starter implementation is prepared for learning and must be reviewed, executed and understood by the student. Run the experiments yourself, inspect the results, write observations in your own words, maintain your own Git commit history, and use your own GitHub and Streamlit accounts. Do not claim deployment or BITS Lab execution until you have actually completed those steps.
