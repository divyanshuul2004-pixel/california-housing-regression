# California Housing Price Prediction: End-to-End Multivariate Regression System

## 🎯 Project Overview & Objective
This project implements a complete, end-to-end machine learning engineering workflow to build a supervised, multivariate regression model predicting a district’s **median house value** using the California Housing Dataset (20,640 records). The core objective was to design a robust production-style automation pipeline that handles raw data ingestion, geographical feature engineering, and hyperparameter optimization while strictly mitigating data snooping bias and under/overfitting patterns.

---

## 🚀 Key Architectural Highlights
*   **Zero Data-Snooping Strategy:** Implemented a deterministic hash-based split utilizing geographical coordinate signatures (`longitude * 1000 + latitude`) to keep train and test sets strictly consistent as new records append.
*   **Stratified Target Splitting:** Leveraged `StratifiedShuffleSplit` on engineered income categories to ensure the training and test sets precisely match the continuous distribution of the crucial `median_income` feature.
*   **Modular Pipeline Architecture:** Built custom object-oriented Scikit-Learn transformers (`CombinedAttributesAdder` and `DataFrameSelector`) coordinated inside a unified parallel `FeatureUnion` track to output an automated 16-feature numerical matrix.
*   **Optimized Performance:** Conducted exhaustive `GridSearchCV` optimization to achieve a final Cross-Validation **RMSE of ~\$50,016** using tree-based ensembles.

---

## 🛠️ Tech Stack & Open-Source Tools
*   **Language:** Python
*   **Data Analysis:** Pandas, NumPy
*   **Machine Learning Framework:** Scikit-Learn (Pipelines, FeatureUnion, Imputers, Encoders, Scalers)
*   **Optimization:** GridSearchCV

---

## ⚙️ Data Preprocessing & Automated Pipeline Layout
The data pipeline splits raw matrices into clean numerical and categorical branches, processing them in parallel before combining them into a final array:

┌──► DataFrameSelector ──► SimpleImputer(Median) ──► CombinedAttributesAdder ──► StandardScaler ┐
Raw Ingested DataFrame ───┤                                                                                                ├──► FeatureUnion ──► Trained Models
└──► DataFrameSelector ──► OneHotEncoder(Sparse=False) ──────────────────────────────────────────┘

1.  **Imputation:** Missing data in `total_bedrooms` is programmatically calculated and resolved using a `SimpleImputer` trained strictly on the training partition.
2.  **Categorical Vectorization:** The `ocean_proximity` feature is converted via `OneHotEncoder` to completely eliminate implicit ordinal hierarchies.
3.  **Custom Feature Ratios:** The pipeline automatically synthesizes domain-specific ratios to boost target correlation:
    *   `rooms_per_household`
    *   `population_per_household`
    *   `bedroom_per_room` (Highly effective inverse target relationship discovered during EDA)

---

## 📊 Cross-Validation Performance Benchmarks
Candidate architectures were thoroughly evaluated using **10-Fold Cross-Validation** to balance bias and variance:

| Model Architecture | Training RMSE | 10-Fold CV Mean RMSE | Validation Std Dev | Evaluation Diagnosis & Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | ~\$68,628 | ~\$69,104 | ~\$2,880 | **Underfitting**: Boundaries are insufficient for non-linear geographical interactions. |
| **Decision Tree Regressor** | \$0.0 | ~\$71,816 | ~\$2,355 | **Severe Overfitting**: Memorized training noise; performed worse on validation sets. |
| **Random Forest Regressor** | ~\$18,651 | **~\$50,436** | ~\$2,203 | **Winning Model**: Captures non-linear feature spaces efficiently; selected for tuning. |
| **Support Vector Regressor (SVR)** | ~\$111,095 | ~\$111,814 | — | **Severe Underfitting**: Default hyperparameter space failed to map the features. |

---

## 📈 Hyperparameter Tuning & Feature Interpretability
Using `GridSearchCV` (5-fold testing over 18 hyperparameter combinations), the maximum features and tree metrics were optimized:
*   **Winning Parameters:** `{'max_features': 6, 'n_estimators': 30}`
*   **Final Optimized Cross-Validation RMSE:** **~\$50,016**

### Top Engineering Predictors
1.  **Median Income (~37.5%)** — Strongest positive linear relationship with housing value.
2.  **Inland Ocean Proximity (~13.6%)** — Found to be the strongest negative location indicator for luxury valuation.

---

## 🔍 System Constraints & Next-Gen Iterations
*   **The Cap Line Artifact:** The target parameter `median_house_value` is artificially capped at \$500,000. Future iterations will involve testing a secondary sub-model specifically designed to predict valuations above this ceiling threshold.
*   **Scale Limits:** While Random Forest proved dominant, training times scale exponentially with broad grids. Next stages include migrating down-stream processes to gradient boosted trees (XGBoost/LightGBM) to improve training latency.

