# House Price Prediction - Regression Project

**Dataset:** King County, Washington Housing Sales (2014)  
**Task:** Predict house sale prices using supervised regression algorithms.

---

## 1. Problem Identification (10 marks)

### Problem Statement
Real estate pricing is complex and influenced by numerous factors including size, location, condition, and amenities. Accurately estimating house prices helps buyers, sellers, and agents make informed decisions. This project builds a machine learning regression model to predict the sale price of houses in King County, WA based on structural and location features.

### Objectives
- Clean and preprocess real-world housing data
- Perform exploratory data analysis to understand price drivers
- Implement and compare multiple regression algorithms
- Tune the best model for improved performance
- Deploy an interactive web application for price prediction

### Success Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (coefficient of determination)

---

## 2. Dataset & Preprocessing (15 marks)

### Source
King County House Sales dataset (publicly available, 2014 transactions).

### Original Features
| Feature | Description |
|---------|-------------|
| date | Sale date |
| price | Sale price (target) |
| bedrooms, bathrooms | Room counts |
| sqft_living, sqft_lot | Size metrics |
| floors | Number of floors |
| waterfront | Waterfront property (0/1) |
| view | View quality (0-4) |
| condition | Overall condition (1-5) |
| sqft_above, sqft_basement | Area breakdown |
| yr_built, yr_renovated | Construction / renovation year |
| street, city, statezip, country | Location |

### Preprocessing Steps
1. **Removed invalid prices**: Dropped 49 records with price = 0 and near-zero values (data errors).
2. **Outlier removal**: Filtered extreme prices (> $5M) and living areas (> 10,000 sqft) → 4,546 clean records.
3. **Feature Engineering**:
   - `age` = year_sold − yr_built
   - `is_renovated` (binary flag)
   - `years_since_reno`
   - `total_rooms` = bedrooms + bathrooms
   - `living_lot_ratio`
   - `has_basement` (binary)
   - Extracted numeric `zipcode` from statezip
4. **Dropped**: street (high cardinality), country (constant = USA), original date/yr_renovated.
5. **Encoding**: One-Hot Encoding for `city`; StandardScaler for all numeric features.
6. **Train/Test Split**: 80/20 stratified by random seed 42.

---

## 3. EDA & Visualization (10 marks)

Key findings (figures saved in `notebooks/figures/`):

- **Price distribution** is right-skewed (mean ≈ $548k, median ≈ $465k). Log-transform helps normality.
- **Strongest predictors** of price:
  1. `sqft_living` (corr ≈ 0.69)
  2. `sqft_above` (0.59)
  3. `bathrooms` (0.53)
  4. `view` (0.38)
- Waterfront properties command significant premiums.
- Higher condition and view ratings correlate with higher prices.
- City-level variation is large (e.g., Medina, Mercer Island, Bellevue vs. more affordable areas).
- Correlation heatmap and scatter/box plots confirm multicollinearity between size-related features (handled by tree models).

---

## 4. ML Algorithm Implementation (20 marks)

Implemented and compared:

| Model | Type |
|-------|------|
| Linear Regression | Baseline parametric |
| Ridge Regression | L2 regularized |
| Lasso Regression | L1 regularized |
| ElasticNet | Combined L1/L2 |
| Random Forest | Ensemble bagging |
| Gradient Boosting | Ensemble boosting |

All models use a scikit-learn `Pipeline` with preprocessing + estimator for reproducibility and to prevent data leakage.

---

## 5. Model Evaluation (10 marks)

### Baseline Results (Test Set)

| Model | Test RMSE | Test MAE | Test R² | CV RMSE |
|-------|-----------|----------|---------|---------|
| Gradient Boosting | $181,417 | $104,346 | **0.715** | $190,866 |
| Ridge | $184,521 | $119,272 | 0.705 | $197,475 |
| Linear / Lasso | ~$184,720 | ~$119,500 | 0.705 | ~$198,300 |
| Random Forest | $191,696 | $108,977 | 0.682 | $199,857 |
| ElasticNet | $201,674 | $131,049 | 0.648 | $213,416 |

Gradient Boosting provided the best balance of accuracy and generalization.

---

## 6. Model Improvement (10 marks)

### Hyperparameter Tuning (GridSearchCV, 3-fold CV)
- **Random Forest**: n_estimators, max_depth, min_samples_leaf
- **Gradient Boosting**: n_estimators, max_depth, learning_rate

**Best tuned model:** Gradient Boosting  
- Params: `n_estimators=150`, `max_depth=4`, `learning_rate=0.1`
- **Improved Test R²: 0.724**
- **Improved Test RMSE: $178,517**
- **Improved Test MAE: $103,773**

Residual plots and actual-vs-predicted diagnostics confirm reasonable error distribution with some remaining heteroscedasticity at high prices.

---

## 7. Application / UI (10 marks)

Interactive **Streamlit** web application (`app/app.py`):

- User-friendly form for all key property attributes
- Real-time price prediction
- Displays model metrics and typical error range
- City dropdown populated from training data
- Derived features (age, ratios, flags) computed automatically

**Run locally:**
```bash
cd house_price_prediction
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 8. GitHub Repository (5 marks)

Recommended structure (this repo):

```
house_price_prediction/
├── data/
│   ├── data.csv              # original
│   └── cleaned_data.csv
├── notebooks/
│   └── figures/              # EDA plots
├── src/
│   └── preprocess_and_train.py
├── models/
│   ├── best_model.joblib
│   └── feature_info.joblib
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

Initialize and push:
```bash
git init
git add .
git commit -m "House price prediction regression project"
git remote add origin <your-repo-url>
git push -u origin main
```

---

## 9. Deployment (5 marks)

### Option A – Streamlit Community Cloud (Recommended)
1. Push repo to GitHub
2. Go to https://share.streamlit.io
3. Connect the repository
4. Set main file path: `app/app.py`
5. Deploy

### Option B – Local / Docker
```bash
streamlit run app/app.py
```

### Option C – Hugging Face Spaces / Render
Use the same Streamlit app; provide `requirements.txt` and the models folder.

---

## 10. Presentation & Viva Tips (5 marks)

Be ready to discuss:
- Why Gradient Boosting outperformed linear models (non-linear relationships, interactions)
- Handling of multicollinearity and high-cardinality city
- Impact of feature engineering (age, renovation, ratios)
- Limitations: 2014 data, no market-cycle features, residual heteroscedasticity
- How you would improve further (XGBoost/LightGBM, target encoding for zip, external economic data)

---

## How to Reproduce

```bash
# Clone / navigate
cd house_price_prediction

# Install
pip install -r requirements.txt

# Train (generates model + figures)
cd src
python preprocess_and_train.py

# Launch UI
cd ../
streamlit run app/app.py
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| Best Model | Tuned Gradient Boosting |
| Test R² | **0.724** |
| Test RMSE | **$178,517** |
| Test MAE | **$103,773** |
| Records used | 4,546 |

---

*Project completed as an end-to-end supervised regression pipeline for educational evaluation.*
