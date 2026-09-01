# House Price Prediction

A professional Machine Learning project using the Ames Housing dataset to predict residential house prices through a complete and reproducible regression workflow.

## Overview

House prices depend on many factors such as property size, overall quality, location-related characteristics, year built, garage features, basement features, and other property attributes.

This project focuses on building a high-performing regression model to predict house `SalePrice`, demonstrating a complete machine learning lifecycle from exploratory data analysis and feature engineering to model selection, hyperparameter tuning, final evaluation, error analysis, and model explainability.

## Dataset

The project uses the **Ames Housing Dataset** from the Kaggle House Prices competition.

**Dataset:**  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The dataset contains **79 explanatory features** describing residential properties in Ames, Iowa.

## Key Features

### 1. Domain-Aware EDA & Preprocessing

- Exploratory analysis of numerical and categorical variables.
- Analysis of target distribution and feature relationships.
- Domain-aware handling of missing values.
- Semantic interpretation of missing values where `NaN` represents the absence of a property feature rather than unknown data.
- Examples include distinguishing between:
  - No pool
  - No garage
  - No basement
  - No fireplace
- Evaluation of target transformations, including raw and `log1p` target representations.

### 2. Feature Engineering

The project introduces **11 domain-specific engineered features** to provide additional information to the regression models.

Examples include:

- `TotalSF`
- `TotalBathrooms`
- `TotalBsmtFinSF`
- `QualArea`
- `HouseAge`
- `RemodAge`
- Garage-related aggregated features
- Other property-level aggregate and age-related features

These engineered variables aim to capture relationships that may not be represented directly by the original dataset features.

### 3. Robust Model Benchmarking

A total of **12 regression models** are benchmarked using **5-fold cross-validation**.

The evaluated models include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)

The primary model-selection metric is **Cross-Validation RMSE**, with MAE and R² also reported.

### 4. Hyperparameter Tuning

The top three models identified during benchmarking are further optimized using `RandomizedSearchCV`.

The tuning stage uses:

- 3-fold cross-validation
- Randomized hyperparameter search
- A controlled number of parameter configurations
- Fixed random state for reproducibility

The tuning process is performed on the top-performing models rather than all models in order to reduce computational cost while focusing optimization on the strongest candidates.

### 5. Final Model Evaluation

After hyperparameter tuning, the selected model is evaluated on a held-out test set.

The final evaluation reports:

- Test RMSE
- Test MAE
- Test R²

The final model is then saved for future use.

### 6. Error Analysis

The project performs detailed residual and prediction-error analysis, including:

- Actual vs. predicted prices
- Residual distributions
- Prediction errors across different price segments
- Identification of properties with unusually large prediction errors
- Analysis of model performance on lower-, middle-, and higher-priced properties

This analysis helps identify where the model performs well and where additional improvements may be required.

### 7. Model Explainability

**SHAP (SHapley Additive exPlanations)** is used to interpret the final model.

The explainability analysis helps identify:

- The most influential features
- Features that increase predicted house prices
- Features that decrease predicted house prices
- The overall contribution of important property characteristics to model predictions

## Workflow

```mermaid
flowchart TD

    A[Raw Dataset] --> B[Domain-Aware EDA]
    B --> C[Missing Value Analysis]
    C --> D[Feature Engineering]
    D --> E[Data Preprocessing Pipeline]
    E --> F[12 Regression Models]
    F --> G[5-Fold Cross Validation]
    G --> H[Select Top 3 Models]
    H --> I[RandomizedSearchCV]
    I --> J[3-Fold Hyperparameter Tuning]
    J --> K[Select Best Tuned Model]
    K --> L[Final Test Evaluation]
    L --> M[Error Analysis]
    M --> N[SHAP Explainability]
    N --> O[Save Model & Results]
````

## Data Preprocessing Pipeline

The preprocessing stage is implemented using a leakage-free combination of `ColumnTransformer` and `Pipeline`.

### Numerical Features

* Median imputation for genuinely missing numerical values.
* `RobustScaler` for feature scaling.
* Robust scaling is used to reduce the influence of extreme values and outliers.

### Categorical Features

* Semantic missing values are interpreted using domain knowledge where appropriate.
* Remaining missing categorical values are imputed using the constant value `"Missing"`.
* Categorical variables are transformed using `OneHotEncoder`.

### Leakage Prevention

All preprocessing transformations are included inside the modeling pipeline.

This ensures that preprocessing parameters are learned only from the training data during cross-validation rather than from the complete dataset.

## Evaluation Metrics

| Metric   | Purpose                                                                              |
| -------- | ------------------------------------------------------------------------------------ |
| **RMSE** | Primary metric for model comparison; penalizes larger prediction errors more heavily |
| **MAE**  | Measures the average absolute prediction error                                       |
| **R²**   | Measures the proportion of variance in house prices explained by the model           |

### Why RMSE?

RMSE is used as the primary model-selection metric because large house-price prediction errors are particularly important in this problem. The squared-error component gives greater weight to larger mistakes.

## Benchmark Results

The initial 5-fold cross-validation benchmark produced the following results:

| Rank | Model                 |    CV MAE |       CV RMSE |      CV R² |
| ---: | --------------------- | --------: | ------------: | ---------: |
|    1 | **Gradient Boosting** | 15,407.17 | **25,963.51** | **0.8846** |
|    2 | HistGradientBoosting  | 16,961.26 |     28,428.82 |     0.8629 |
|    3 | XGBoost               | 17,198.61 |     28,466.40 |     0.8628 |
|    4 | Extra Trees           | 16,884.68 |     28,812.91 |     0.8584 |
|    5 | Random Forest         | 17,619.83 |     29,353.30 |     0.8535 |
|    6 | Ridge                 | 18,064.98 |     31,978.74 |     0.8234 |
|    7 | ElasticNet            | 19,556.83 |     34,528.45 |     0.7971 |
|    8 | Lasso                 | 18,675.48 |     37,827.87 |     0.7467 |
|    9 | Linear Regression     | 18,907.33 |     39,600.41 |     0.7141 |
|   10 | Decision Tree         | 26,351.73 |     42,267.14 |     0.6973 |
|   11 | KNN                   | 29,450.79 |     48,677.66 |     0.5960 |
|   12 | SVR                   | 54,585.36 |     78,859.57 |    -0.0459 |

### Initial Benchmark Interpretation

The initial benchmark identified **Gradient Boosting** as the strongest model before hyperparameter tuning, achieving:

* CV RMSE: **$25,963.51**
* CV MAE: **$15,407.17**
* CV R²: **0.8846**

The three highest-performing models according to CV RMSE were selected for further hyperparameter tuning:

1. Gradient Boosting
2. HistGradientBoosting
3. XGBoost

> **Note:** These are the initial cross-validation benchmark results. Final model performance should be reported separately using the completed tuning and held-out test evaluation.

## Hyperparameter Tuning Strategy

The top three models are optimized using `RandomizedSearchCV`.

The tuning stage is intentionally smaller than the initial model benchmark to provide a practical balance between computational cost and model optimization.

```text
Initial Benchmark
        │
        ▼
12 Models
        │
        ▼
5-Fold Cross-Validation
        │
        ▼
Top 3 Models
        │
        ▼
RandomizedSearchCV
        │
        ▼
3-Fold CV
        │
        ▼
Best Tuned Model
```

## Model Selection

The final model is selected based on the lowest cross-validation RMSE obtained during the hyperparameter-tuning stage.

The selected estimator is then evaluated once on the held-out test set to estimate its final generalization performance.

## Error Analysis

Model performance is investigated beyond aggregate metrics.

The error-analysis stage examines:

* Residual distributions
* Actual vs. predicted values
* Absolute prediction errors
* Errors across house-price segments
* High-error observations
* Model behavior on expensive properties

A particular focus is placed on high-value properties because large-price homes can produce substantially larger absolute prediction errors.

## Model Explainability

SHAP is used to provide model-level and prediction-level explanations.

The analysis focuses on understanding which property characteristics contribute most strongly to predicted `SalePrice`.

This provides a more interpretable view of the model beyond traditional performance metrics.

## Project Structure

```text
house-price-ml/
│
├── data/
│   └── train.csv
│
├── notebooks/
│   └── house_price_ml.ipynb
│
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
│
├── reports/
│   ├── model_results.csv
│   └── figures/
│       ├── correlation_analysis.png
│       ├── missing_values.png
│       ├── model_comparison.png
│       └── target_distribution.png
│
├── README.md
└── requirements.txt
```

> Additional generated figures or model artifacts may be added to these directories as the project evolves.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/house-price-ml.git
cd house-price-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Activate the Environment on Windows

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/house_price_ml.ipynb
```

Run the notebook cells in order.

The notebook automatically generates the required model results, figures, and saved model artifacts in the `reports/` and `models/` directories.

## Outputs

The project generates:

### Models

```text
models/best_model.joblib
models/model_metadata.json
```

### Reports

```text
reports/model_results.csv
```

### Figures

```text
reports/figures/
```

The generated figures include exploratory-analysis and model-comparison visualizations.

