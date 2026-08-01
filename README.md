# Stroke Risk Prediction: A Data Analytics Case Study

## Business Task

Identify which health and lifestyle factors most strongly predict stroke risk, in order to inform early-warning awareness for at-risk individuals.

**Stakeholders:** Public health educators and at-risk individuals who could use this analysis to understand which factors are most worth monitoring.

**Guiding question:** Can we predict stroke risk from health/lifestyle features, and which factors matter most?

## About This Project

Stroke is a leading cause of long-term disability, and much of the risk is tied to a handful of measurable health and lifestyle factors. This project uses a real-world clinical dataset to explore which of those factors are most strongly associated with stroke, and to build a predictive model that could support early-warning efforts.

## Data Source

- **Dataset:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) by fedesoriano, via Kaggle
- **Size:** 5,110 records, 12 columns
- **License:** See Kaggle page for terms. Raw data is not redistributed in this repo — see `data/README.md` for the source link.

**Features:** gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status, and stroke occurrence (target).

## Methodology (Ask → Act Framework)

### Ask
Defined the business task above and selected a predictive modeling approach (rather than a purely descriptive one) to surface which factors matter most, not just how they differ on average.

### Prepare
- Verified data credibility (well-documented, ~300K downloads, widely used in published research)
- Checked for licensing/access constraints
- Confirmed the dataset is a single snapshot per patient (not longitudinal) — this ruled out an initial "change over time" framing in favor of the predictive question above

### Process (Data Cleaning)
- Imputed 201 missing `bmi` values using the median (chosen over mean because BMI is right-skewed; median is resistant to outlier influence)
- Removed 1 row where `gender = "Other"` — a single-sample category not viable for train/test splitting or statistical inference
- Dropped `id` — a row identifier with no predictive value
- Verified no duplicate records
- Kept "Unknown" as a valid category within `smoking_status` (~30% of rows) rather than dropping or imputing it, since it represents a meaningful share of the data

### Analyze
- **Exploratory analysis:** distribution and stroke-rate breakdowns across all categorical and numeric features
- **Statistical testing:**
  - Independent t-tests (numeric features vs. stroke): age, average glucose level, and BMI were all significantly associated with stroke (p < 0.001, p < 0.001, p = 0.002 respectively)
  - Chi-square tests (categorical features vs. stroke): hypertension, heart disease, marital status, work type, and smoking status were all significantly associated with stroke (p < 0.001 to p < 0.001); gender and residence type were **not** statistically significant (p = 0.53, p = 0.28)
- **Modeling approach:**
  - One-hot encoded categorical variables
  - Standardized numeric features (age, glucose, BMI) so logistic regression coefficients would be comparable across features
  - Split data 80/20 (train/test), stratified to preserve the ~4.9% stroke rate in both sets
  - Applied SMOTE to the training data only, to address severe class imbalance (only 199 stroke cases in the training set)
  - Trained and compared two models: **Logistic Regression** and **Random Forest**

### Share (Key Findings)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 76.6% | 90.0% |
| Precision | 0.146 | 0.190 |
| **Recall** | **0.78** | 0.32 |
| F1 Score | 0.246 | 0.239 |
| ROC-AUC | 0.833 | 0.775 |

**Random Forest has higher raw accuracy, but Logistic Regression is the better model for this task.** Because stroke cases are rare (~5% of the data), a model can score high on accuracy while still missing most real stroke cases. Since the goal is early-warning awareness, **recall** — the share of actual stroke cases the model successfully flags — matters more than overall accuracy. Logistic Regression caught 78% of stroke cases in the test set versus 32% for Random Forest.

**Strongest predictors of stroke** (consistent across both models and the statistical tests): **age, average glucose level, hypertension,** and **BMI**. Marital status also showed a strong association, but this is very likely a proxy for age (married individuals in this dataset skew older) rather than an independent effect.

### Limitations
- **Small positive class:** only 249 stroke cases across the full dataset (5,110 rows), which limits how much any model can learn about a rare outcome — SMOTE can help balance training but can't manufacture genuinely new information.
- **Missing risk factors:** the dataset doesn't include family history, cholesterol, blood pressure readings, physical activity, or prior TIA/mini-stroke history — all known real-world stroke risk factors not captured here.
- **Multicollinearity artifact:** the logistic regression coefficient for `work_type_children` appeared unusually large. This is not a genuine finding — `work_type_children` is almost entirely a proxy for very young age (ages 0.08–16 in this dataset), so it overlaps heavily with the `age` variable. The coefficient math splits credit unpredictably between overlapping features; age itself (confirmed independently by the t-test and Random Forest's feature importance) is the reliable signal here, not the work-type category.
- **Correlation, not causation:** all findings describe statistical association, not proven biological cause.

### Act (Recommendations)
- Early-warning efforts should prioritize monitoring for individuals who are older, have hypertension, and/or have elevated average glucose levels — the most consistent predictors across both statistical testing and modeling.
- Given the model's limitations, this analysis is best used as a starting point for risk awareness, not a diagnostic tool. A more clinically complete dataset (with blood pressure readings, cholesterol, family history) would likely improve predictive performance meaningfully.
- Future analysis could explore additional models (e.g. gradient boosting) or feature engineering (e.g. age-hypertension interaction terms) to address the precision/recall tradeoff currently limiting the model's practical usefulness.

### Summary 

Business Question: Can we predict stroke risk from health/lifestyle features, and which factors matter most?

Approach: After cleaning the data (median-imputed BMI, removed a single "Other" gender row, dropped id), I ran independent t-tests and chi-square tests to check which variables were statistically associated with stroke, then built two predictive models — Logistic Regression and Random Forest — after encoding categorical variables, standardizing numeric features, and applying SMOTE to address the ~95%/5% class imbalance.

Key Finding #1 — Which factors matter most:
Age, average glucose level, hypertension, and BMI were consistently identified as the strongest predictors across three independent methods: statistical testing (t-test/chi-square), Random Forest's feature importance ranking, and (mostly) Logistic Regression's coefficients. This convergence across different methods gives confidence these are genuine patterns in the data.

Key Finding #2 — Model comparison:
Random Forest achieved higher raw accuracy (90% vs. 76.6%), but Logistic Regression achieved much higher recall (78% vs. 32%) and ROC-AUC (0.83 vs. 0.78). Since accuracy is misleading on an imbalanced dataset (a model predicting "no stroke" for everyone would already score ~95% accuracy), recall is the more meaningful metric for this early-warning use case — missing an actual stroke case is more costly than a false alarm. I re-ran Logistic Regression across 30 random train/test splits to confirm this performance was stable and not a result of one lucky split; recall and ROC-AUC stayed consistent across all 30 runs (recall mean ≈ 0.71, std ≈ 0.06).

Key Finding #3 — A data-quality limitation I identified and diagnosed:
Logistic Regression's coefficient for work_type_children appeared unusually large and statistically significant, contradicting both the stroke-rate chart (children ≈ 0% stroke rate) and Random Forest's importance ranking (near-zero for this feature). I traced this to its root cause: the entire dataset contains only one confirmed stroke case among 555 patients labeled work_type = children. Re-running the model across 30 random splits showed the coefficient was strongly positive whenever that single row landed in the training set, and flipped negative in the 3 runs where it didn't — a bimodal pattern directly caused by one data point's placement. This demonstrates that Maximum Likelihood Estimation can produce confident-looking coefficients even from statistically unsupported subgroups, and shows why a coefficient's size or p-value alone isn't sufficient evidence of a real predictive relationship — sample size and cross-method agreement matter too.

Conclusion / Recommendation:
Early-warning efforts should prioritize age, glucose levels, and hypertension as the most reliable risk indicators. The dataset's small number of total stroke cases (249 of 5,110) limits how much confidence can be placed in findings for smaller subgroups, and a larger, more clinically complete dataset (including blood pressure readings, cholesterol, and family history) would likely improve both model performance and the reliability of subgroup-level findings.

## Repository Structure

```
stroke-risk-capstone/
├── README.md              This file — full write-up of the analysis
├── data/                  Note on data source (raw data not redistributed)
├── notebooks/
│   └── analysis.ipynb     Full analysis: cleaning, EDA, stats, modeling, evaluation
├── images/                Exported charts and visualizations
└── requirements.txt       Python dependencies
```

## Tools Used
Python (pandas, numpy, seaborn, matplotlib, scipy, statsmodels, scikit-learn, imbalanced-learn)

## Acknowledgements
Dataset provided by fedesoriano on Kaggle. This project was completed as the capstone case study for the Google Data Analytics Professional Certificate.
