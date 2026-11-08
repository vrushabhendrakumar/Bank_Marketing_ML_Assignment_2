# Bank Marketing Classification Model Lab

**BITS Pilani WILP | M.Tech (DSE) | Machine Learning | Assignment 2**

**Student:** Vrushabhendra Kumar K N  
**BITS ID:** 202NS04021

## a. Problem statement

The objective is to predict whether a client of a Portuguese banking institution subscribes to a term deposit after direct-marketing activity. This is a binary classification task with `y` as the target (`yes` or `no`).

Five classification algorithms were trained and evaluated: Logistic Regression, Decision Tree, k-Nearest Neighbors (kNN), Gaussian Naive Bayes and Support Vector Machine (LinearSVC). Each model was evaluated on the same stratified 20% test split using Accuracy, ROC-AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

**Leakage prevention:** The `duration` feature was excluded. Call duration is only known after a call has ended, so including it would not be appropriate for a model intended to support decisions before or during a call.

## b. Dataset description

- **Dataset:** Bank Marketing (`bank-full.csv`, legacy version).
- **Source:** UCI Machine Learning Repository, dataset ID 222.
- **Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- **Official dataset page:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Number of records:** 45,211.
- **Original input features:** 16.
- **Predictors used:** 15, after excluding `duration`.
- **Target:** `y`, indicating whether the client subscribed to a term deposit (`yes` or `no`).
- **Data types:** A mixture of numeric and categorical features.
- **Class distribution in the held-out test set:** 7,985 `no` records and 1,058 `yes` records. The target is imbalanced, with `yes` as the minority class.

## c. GitHub Repository Link

**Repository:** https://github.com/vrushabhendrakumar/Bank_Marketing_ML_Assignment_2

The repository contains the project source, training code, notebook, application, dependencies and experiment result files. The repository link should be opened to inspect the current committed files.

## Live Streamlit application

**Application:** https://bankmarketingmlassignment2-cda2zs3rb6toynwhz5f755.streamlit.app/

The application is intended to provide a CSV upload option, a model-selection dropdown, evaluation metrics and a confusion matrix/classification report for labelled test data. The public URL should be tested in a normal browser or private window before submission to confirm that the interactive app loads.

## d. Models used

1. **Logistic Regression:** A linear classification model. Class weighting was enabled.
2. **Decision Tree:** A tree-based classifier that can represent non-linear decision rules.
3. **k-Nearest Neighbors:** A distance-based classifier using 7 neighbours with distance weighting.
4. **Gaussian Naive Bayes:** A probabilistic classifier based on Gaussian likelihoods and a conditional-independence assumption.
5. **Support Vector Machine:** LinearSVC with class weighting. Decision scores were used to calculate ROC-AUC.

### Experiment setup and preprocessing

- Train/test split: 80% / 20%, stratified by target.
- Random state: 42.
- Dataset size: 45,211 rows; held-out test size: 9,043 rows.
- Numeric features: median imputation followed by standard scaling.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Preprocessing was fitted on the training data as part of scikit-learn pipelines.
- `duration` was excluded to avoid leakage.
- Class weighting was applied to Logistic Regression, Decision Tree and LinearSVC.
- ROC-AUC was calculated from predicted probabilities where available and decision scores for LinearSVC.

### Model comparison table

All values below are from the supplied experiment output (`results/model_comparison.csv`) and are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7622 | 0.7518 | 0.2681 | 0.5964 | 0.3699 | 0.2788 |\n| Decision Tree | 0.8330 | 0.7384 | 0.3507 | 0.5019 | 0.4129 | 0.3261 |\n| K-Nearest Neighbors (kNN) | 0.8888 | 0.7104 | 0.5730 | 0.1928 | 0.2885 | 0.2872 |\n| Gaussian Naive Bayes | 0.8466 | 0.7340 | 0.3689 | 0.4376 | 0.4003 | 0.3146 |\n| Support Vector Machine (LinearSVC) | 0.7687 | 0.7513 | 0.2732 | 0.5888 | 0.3733 | 0.2823 |\n
### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It achieved the highest recall (0.5964) and the highest ROC-AUC (0.7518). It identified more subscribers than most other models, but its precision was relatively low (0.2681), so a substantial number of predicted subscribers were false positives. |\n| Decision Tree | It produced the highest F1-score (0.4129) and MCC (0.3261), giving the strongest balance of precision and recall among the five models. Its accuracy was 0.8330 and its recall was 0.5019. |\n| kNN | It recorded the highest accuracy (0.8888) and precision (0.5730), but recall was only 0.1928. It missed many actual subscribers, so its high accuracy does not make it the best model for detecting the minority class. |\n| Naive Bayes | Gaussian Naive Bayes gave a reasonable baseline, with F1-score 0.4003 and MCC 0.3146. Its performance is useful for comparison, although the conditional-independence assumption may not fully describe relationships among the bank-marketing features. |\n| Support Vector Machine | The LinearSVC model achieved ROC-AUC 0.7513 and recall 0.5888, close to Logistic Regression. Precision was 0.2732 and F1-score was 0.3733, indicating a similar precision-recall trade-off. |\n| **Overall winner for this dataset** | **Decision Tree**, when selecting on the combined balance of F1-score and MCC. It achieved the highest F1-score (0.4129) and MCC (0.3261). Logistic Regression had the highest recall and AUC, so it may be preferable if identifying as many potential subscribers as possible is the main priority. |

### Interpretation

The test set contains 7,985 non-subscribers and 1,058 subscribers. A classifier that always predicted `no` would achieve approximately 0.8830 accuracy. The kNN accuracy of 0.8888 is only slightly higher than that reference, while its recall is 0.1928. This shows why accuracy alone is not sufficient for this imbalanced classification task.

The Decision Tree is selected as the overall winner based on F1-score and MCC, which provide a more balanced view of positive-class detection and overall prediction quality. The best model may differ if the practical objective places greater cost on false negatives or false positives.

## Streamlit application features

The submitted application is expected to provide:

- CSV upload for labelled test data.
- A dropdown to select one of the five trained models.
- Accuracy, ROC-AUC, Precision, Recall, F1-score and MCC.
- A confusion matrix and/or classification report.
- A view of actual and predicted target labels.

## Project structure

```text
Bank_Marketing_ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── download_dataset.py
├── model/
│   ├── train_models.py
│   └── saved_models/
├── results/
│   ├── model_comparison.csv
│   ├── evaluation_details.json
│   └── run_metadata.json
└── notebooks/
    └── Bank_Marketing_Classification.ipynb
```

## Installation and execution

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python data/download_dataset.py
python model/train_models.py
streamlit run app.py
```

After training, the script generates the saved model pipelines, held-out `test_data.csv` and result files under `results/`. The labelled test CSV can then be uploaded in the Streamlit application.

## Streamlit Community Cloud deployment

The app is configured for deployment from the GitHub repository using Streamlit Community Cloud. The entry point is `app.py`, and the dependencies are listed in `requirements.txt`.

Deployment URL: https://bankmarketingmlassignment2-cda2zs3rb6toynwhz5f755.streamlit.app/

## Screenshots to insert before final submission

Add genuine screenshots in the spaces provided in the accompanying report:
1. Streamlit application home page / CSV upload interface.
2. Streamlit model selection and displayed evaluation metrics.
3. Streamlit confusion matrix or classification report.
