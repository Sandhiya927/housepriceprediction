"""
House Price Prediction Web App
Streamlit UI for the trained Gradient Boosting model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="King County House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Paths
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.joblib"
FEATURE_INFO_PATH = Path(__file__).parent.parent / "models" / "feature_info.joblib"

@st.cache_resource
def load_model_and_info():
    model = joblib.load(MODEL_PATH)
    info = joblib.load(FEATURE_INFO_PATH)
    return model, info

def main():
    st.title("🏠 King County House Price Predictor")
    st.markdown("""
    Predict house prices in King County, Washington using a tuned **Gradient Boosting** regression model.
    Data from 2014 real estate transactions.
    """)
    
    try:
        model, info = load_model_and_info()
    except Exception as e:
        st.error(f"Could not load model: {e}. Please run the training script first.")
        return
    
    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"**Model:** {info['model_name']}")
    st.sidebar.markdown(f"**Test R²:** {info['metrics']['r2']:.4f}")
    st.sidebar.markdown(f"**Test RMSE:** ${info['metrics']['rmse']:,.0f}")
    st.sidebar.markdown(f"**Test MAE:** ${info['metrics']['mae']:,.0f}")
    
    st.header("Enter Property Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=8.0, value=2.0, step=0.25)
        sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=10000, value=1800, step=50)
        sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=500000, value=7000, step=100)
        floors = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=0)
    
    with col2:
        waterfront = st.selectbox("Waterfront", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        view = st.slider("View Rating (0-4)", min_value=0, max_value=4, value=0)
        condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
        sqft_above = st.number_input("Above Ground Sqft", min_value=300, max_value=10000, value=1500, step=50)
        sqft_basement = st.number_input("Basement Sqft", min_value=0, max_value=5000, value=0, step=50)
    
    with col3:
        yr_built = st.number_input("Year Built", min_value=1900, max_value=2015, value=1980, step=1)
        city = st.selectbox("City", options=info['cities'], index=info['cities'].index('Seattle') if 'Seattle' in info['cities'] else 0)
        month_sold = st.slider("Month of Sale", min_value=1, max_value=12, value=6)
        is_renovated = st.selectbox("Renovated?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        zipcode = st.number_input("Zip Code", min_value=98000, max_value=98200, value=98103, step=1)
    
    # Derived features (must match training)
    age = 2014 - yr_built  # approximate as training used year_sold ~2014
    years_since_reno = age if is_renovated == 0 else max(0, 2014 - (yr_built + 10))  # rough
    total_rooms = bedrooms + bathrooms
    living_lot_ratio = sqft_living / (sqft_lot + 1)
    has_basement = 1 if sqft_basement > 0 else 0
    
    # Build input dataframe with correct column order
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'sqft_lot': [sqft_lot],
        'floors': [floors],
        'waterfront': [waterfront],
        'view': [view],
        'condition': [condition],
        'sqft_above': [sqft_above],
        'sqft_basement': [sqft_basement],
        'yr_built': [yr_built],
        'city': [city],
        'month_sold': [month_sold],
        'age': [age],
        'is_renovated': [is_renovated],
        'years_since_reno': [years_since_reno],
        'total_rooms': [total_rooms],
        'living_lot_ratio': [living_lot_ratio],
        'has_basement': [has_basement],
        'zipcode': [zipcode]
    })
    
    st.markdown("---")
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        try:
            prediction = model.predict(input_data)[0]
            
            st.success(f"### Estimated House Price: **${prediction:,.0f}**")
            
            # Confidence-ish range based on MAE
            mae = info['metrics']['mae']
            st.info(f"Typical error range (based on model MAE): ${prediction - mae:,.0f} — ${prediction + mae:,.0f}")
            
            # Simple interpretation
            st.subheader("Key Factors")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Living Area", f"{sqft_living:,} sqft")
            col_b.metric("Bedrooms / Baths", f"{bedrooms} / {bathrooms}")
            col_c.metric("Age", f"{age} years")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    
    st.markdown("---")
    st.subheader("About the Model")
    st.markdown("""
    - **Algorithm:** Gradient Boosting Regressor (tuned via GridSearchCV)
    - **Features:** Size metrics, location (city + zip), condition, view, waterfront, age, renovation status, etc.
    - **Preprocessing:** Standard scaling for numeric features + One-Hot Encoding for city
    - **Performance:** ~72% of variance explained (R² ≈ 0.72) on held-out test set
    - **Data:** ~4,500 cleaned records from King County, WA (2014)
    
    **Note:** This is an educational model. Real estate prices depend on many market factors not captured here.
    """)

if __name__ == "__main__":
    main()
