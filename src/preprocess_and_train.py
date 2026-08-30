"""
House Price Prediction - Regression Project
King County Housing Dataset (2014)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_clean_data(path='../data/data.csv'):
    """Load and clean the dataset"""
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")
    
    # Drop zero or near-zero prices (data errors)
    df = df[df['price'] > 10000].copy()
    print(f"After removing invalid prices: {df.shape}")
    
    # Drop extreme outliers (price > 5M or sqft_living > 10000)
    df = df[(df['price'] < 5_000_000) & (df['sqft_living'] < 10000)].copy()
    print(f"After outlier removal: {df.shape}")
    
    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['year_sold'] = df['date'].dt.year
    df['month_sold'] = df['date'].dt.month
    
    # Age of house at sale
    df['age'] = df['year_sold'] - df['yr_built']
    
    # Renovation flag and years since reno
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['years_since_reno'] = np.where(
        df['yr_renovated'] > 0,
        df['year_sold'] - df['yr_renovated'],
        df['age']  # if never renovated, use age
    )
    
    # Total rooms approx
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Living area ratio
    df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
    
    # Basement flag
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    
    # Extract zip code
    df['zipcode'] = df['statezip'].str.extract(r'(\d{5})').astype(int)
    
    # Drop unused columns
    drop_cols = ['date', 'street', 'statezip', 'country', 'yr_renovated', 'year_sold']
    df = df.drop(columns=drop_cols)
    
    print(f"Final shape after feature engineering: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def perform_eda(df, save_dir='../notebooks/figures'):
    """Generate EDA visualizations"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Price distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title('Price Distribution')
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_title('Log(Price) Distribution')
    axes[1].set_xlabel('Log(Price)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap (numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Price vs key features
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features = ['sqft_living', 'bedrooms', 'bathrooms', 'age', 'condition', 'view']
    for ax, feat in zip(axes.flat, features):
        if feat in ['bedrooms', 'condition', 'view']:
            sns.boxplot(data=df, x=feat, y='price', ax=ax)
        else:
            ax.scatter(df[feat], df['price'], alpha=0.3, s=10)
        ax.set_title(f'Price vs {feat}')
        ax.set_ylabel('Price')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/price_vs_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. City-wise average price (top 15)
    city_price = df.groupby('city')['price'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    city_price.plot(kind='barh', color='teal')
    plt.title('Average House Price by City (Top 15)')
    plt.xlabel('Average Price ($)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/city_avg_price.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Floors and waterfront effect
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x='floors', y='price', ax=axes[0])
    axes[0].set_title('Price by Number of Floors')
    sns.boxplot(data=df, x='waterfront', y='price', ax=axes[1])
    axes[1].set_title('Price by Waterfront (0/1)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/floors_waterfront.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"EDA figures saved to {save_dir}")
    
    # Print key stats
    print("\n=== Key Statistics ===")
    print(f"Mean price: ${df['price'].mean():,.0f}")
    print(f"Median price: ${df['price'].median():,.0f}")
    print(f"Std price: ${df['price'].std():,.0f}")
    print(f"\nTop correlations with price:")
    print(corr['price'].sort_values(ascending=False).head(10))


def prepare_features(df):
    """Prepare X, y and define preprocessing"""
    # Target
    y = df['price']
    
    # Features to use
    # Drop high-cardinality or leaky
    feature_cols = [c for c in df.columns if c not in ['price', 'city']]  # city will be encoded separately or dropped for simplicity
    
    # Actually keep city as categorical
    X = df.drop(columns=['price'])
    
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = ['city']  # only city is object now
    
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    return X, y, preprocessor, numeric_features, categorical_features


def train_and_evaluate_models(X, y, preprocessor):
    """Train multiple models and evaluate"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    results = []
    best_model = None
    best_score = -np.inf
    best_name = None
    trained_pipelines = {}
    
    print("\n=== Model Training & Evaluation ===\n")
    
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit
        pipe.fit(X_train, y_train)
        
        # Predict
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # CV
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, 
                                     scoring='neg_root_mean_squared_error', n_jobs=-1)
        cv_rmse = -cv_scores.mean()
        
        results.append({
            'Model': name,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R2': test_r2,
            'CV RMSE': cv_rmse
        })
        
        trained_pipelines[name] = pipe
        
        print(f"{name}:")
        print(f"  Test RMSE: ${test_rmse:,.0f} | MAE: ${test_mae:,.0f} | R²: {test_r2:.4f} | CV RMSE: ${cv_rmse:,.0f}")
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model = pipe
            best_name = name
    
    results_df = pd.DataFrame(results).sort_values('Test R2', ascending=False)
    print("\n=== Results Summary ===")
    print(results_df.to_string(index=False))
    
    print(f"\nBest model: {best_name} with R² = {best_score:.4f}")
    
    return results_df, best_model, best_name, X_train, X_test, y_train, y_test, trained_pipelines


def improve_model(X, y, preprocessor, X_train, y_train):
    """Hyperparameter tuning for best models"""
    print("\n=== Model Improvement: Hyperparameter Tuning ===\n")
    
    # Tune Random Forest
    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    # Lightweight tuning for speed
    rf_param_grid = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [12, 18],
        'model__min_samples_leaf': [1, 2]
    }
    
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=3, 
                           scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    
    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Best RF CV RMSE: ${-rf_grid.best_score_:,.0f}")
    
    # Tune Gradient Boosting (smaller grid)
    gb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(random_state=42))
    ])
    
    gb_param_grid = {
        'model__n_estimators': [100, 150],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.05, 0.1]
    }
    
    gb_grid = GridSearchCV(gb_pipe, gb_param_grid, cv=3,
                           scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=0)
    gb_grid.fit(X_train, y_train)
    
    print(f"Best GB params: {gb_grid.best_params_}")
    print(f"Best GB CV RMSE: ${-gb_grid.best_score_:,.0f}")
    
    # Choose the better one
    if -rf_grid.best_score_ < -gb_grid.best_score_:
        best_tuned = rf_grid.best_estimator_
        best_tuned_name = "Tuned Random Forest"
        best_cv = -rf_grid.best_score_
    else:
        best_tuned = gb_grid.best_estimator_
        best_tuned_name = "Tuned Gradient Boosting"
        best_cv = -gb_grid.best_score_
    
    print(f"\nSelected improved model: {best_tuned_name} (CV RMSE: ${best_cv:,.0f})")
    
    return best_tuned, best_tuned_name


def evaluate_final(model, X_test, y_test, name):
    """Final evaluation"""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== Final Evaluation: {name} ===")
    print(f"Test RMSE: ${rmse:,.0f}")
    print(f"Test MAE:  ${mae:,.0f}")
    print(f"Test R²:   {r2:.4f}")
    
    # Residual plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Price')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    
    axes[1].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1].set_xlabel('Actual Price')
    axes[1].set_ylabel('Predicted Price')
    axes[1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    plt.savefig('../notebooks/figures/final_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    # 1. Load & clean
    df = load_and_clean_data()
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Prepare
    X, y, preprocessor, num_feats, cat_feats = prepare_features(df)
    
    # 4. Train baseline models
    results_df, best_model, best_name, X_train, X_test, y_train, y_test, pipelines = train_and_evaluate_models(
        X, y, preprocessor
    )
    
    # 5. Improve
    improved_model, improved_name = improve_model(X, y, preprocessor, X_train, y_train)
    
    # Evaluate improved
    final_metrics = evaluate_final(improved_model, X_test, y_test, improved_name)
    
    # Also evaluate the original best for comparison
    evaluate_final(best_model, X_test, y_test, best_name)
    
    # Save the best overall model (improved)
    joblib.dump(improved_model, '../models/best_model.joblib')
    print("\nSaved best model to ../models/best_model.joblib")
    
    # Save feature info for the app
    feature_info = {
        'numeric_features': num_feats,
        'categorical_features': cat_feats,
        'cities': sorted(X['city'].unique().tolist()),
        'model_name': improved_name,
        'metrics': final_metrics
    }
    joblib.dump(feature_info, '../models/feature_info.joblib')
    
    # Save cleaned data sample for reference
    df.to_csv('../data/cleaned_data.csv', index=False)
    
    print("\n=== Project Complete ===")
    return results_df, final_metrics


if __name__ == '__main__':
    main()
