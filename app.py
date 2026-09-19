import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------------
# 1. Model Loading with Caching
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model(model_path: str):
    """Load and cache the trained joblib model pipeline."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

@st.cache_data
def load_metadata(metadata_path: str):
    """Load model performance metadata if available."""
    p = Path(metadata_path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None

model = load_trained_model("models/best_model.joblib")
metadata = load_metadata("models/model_metadata.json")

# ---------------------------------------------------------
# 2. Feature Engineering Helper Function
# ---------------------------------------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the 11 engineered features from raw input data."""
    df = df.copy()
    
    # Area features
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (
        df["WoodDeckSF"] + df["OpenPorchSF"] + 
        df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    
    # Bathroom counts
    df["TotalBathrooms"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] + 
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    
    # Property age
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    
    # Binary status indicators
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    
    # Quality-Area Interaction
    df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
    
    return df

# ---------------------------------------------------------
# 3. Default Dataset Columns Blueprint
# ---------------------------------------------------------
def get_default_row() -> dict:
    """Returns a full single-row dictionary containing default Ames dataset values."""
    return {
        'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 68.0, 'LotArea': 9500,
        'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl',
        'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl',
        'Neighborhood': 'NAmes', 'Condition1': 'Norm', 'Condition2': 'Norm',
        'BldgType': '1Fam', 'HouseStyle': '1Story', 'OverallQual': 6,
        'OverallCond': 5, 'YearBuilt': 1973, 'YearRemodAdd': 1994,
        'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd', 'MasVnrType': 'None', 'MasVnrArea': 0.0,
        'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc',
        'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No',
        'BsmtFinType1': 'Unf', 'BsmtFinSF1': 0.0, 'BsmtFinType2': 'Unf',
        'BsmtFinSF2': 0.0, 'BsmtUnfSF': 900.0, 'TotalBsmtSF': 900.0,
        'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y',
        'Electrical': 'SBrkr', '1stFlrSF': 1000.0, '2ndFlrSF': 500.0,
        'LowQualFinSF': 0.0, 'GrLivArea': 1500.0, 'BsmtFullBath': 0.0,
        'BsmtHalfBath': 0.0, 'FullBath': 2, 'HalfBath': 1,
        'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'TA',
        'TotRmsAbvGrd': 6, 'Functional': 'Typ', 'Fireplaces': 1,
        'FireplaceQu': 'TA', 'GarageType': 'Attchd', 'GarageYrBlt': 1973.0,
        'GarageFinish': 'RFn', 'GarageCars': 2, 'GarageArea': 480.0,
        'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y',
        'WoodDeckSF': 0.0, 'OpenPorchSF': 40.0, 'EnclosedPorch': 0.0,
        '3SsnPorch': 0.0, 'ScreenPorch': 0.0, 'PoolArea': 0.0,
        'PoolQC': 'None', 'Fence': 'None', 'MiscFeature': 'None',
        'MiscVal': 0, 'MoSold': 6, 'YrSold': 2008, 'SaleType': 'WD',
        'SaleCondition': 'Normal'
    }

# ---------------------------------------------------------
# 4. Streamlit UI Components
# ---------------------------------------------------------
st.title("🏠 Ames House Price Prediction")
st.markdown("""
Estimate home values using a Machine Learning model trained on the **Ames Housing Dataset**. 
Adjust property characteristics in the sidebar to generate a real-time valuation.
""")

st.sidebar.header("📋 House Characteristics")

# Form inputs grouped into expanders
with st.sidebar.expander("📍 Location & Basic Info", expanded=True):
    neighborhoods = [
        'NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert',
        'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor',
        'Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'SWISU',
        'StoneBr', 'Blmngtn', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'
    ]
    neighborhood = st.selectbox("Neighborhood", options=sorted(neighborhoods), index=neighborhoods.index('NAmes'))
    overall_qual = st.slider("Overall Quality (1=Very Poor, 10=Very Excellent)", 1, 10, 6)
    year_built = st.slider("Year Built", 1872, 2010, 1973)
    year_remod = st.slider("Year Remodeled", 1950, 2010, 1994)
    yr_sold = st.selectbox("Year Sold", [2006, 2007, 2008, 2009, 2010], index=2)

with st.sidebar.expander("📐 Size & Layout", expanded=False):
    gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=300, max_value=6000, value=1500, step=50)
    flr_1st_sf = st.number_input("1st Floor Area (sq ft)", min_value=300, max_value=5000, value=1000, step=50)
    flr_2nd_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, max_value=3000, value=500, step=50)
    tot_rooms = st.number_input("Total Rooms Above Grade", min_value=2, max_value=15, value=6)

with st.sidebar.expander("🛋️ Bathrooms & Amenities", expanded=False):
    full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
    half_bath = st.number_input("Half Bathrooms", 0, 3, 1)
    bsmt_full_bath = st.number_input("Basement Full Bathrooms", 0, 3, 0)
    bsmt_half_bath = st.number_input("Basement Half Bathrooms", 0, 2, 0)
    fireplaces = st.number_input("Fireplaces", 0, 4, 1)
    kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)
    exter_qual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa", "Po"], index=2)

with st.sidebar.expander("🚗 Basement, Garage & Outdoor", expanded=False):
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 4000, 900, step=50)
    garage_cars = st.number_input("Garage Car Capacity", 0, 5, 2)
    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 480, step=50)
    wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", 0, 1000, 0, step=25)
    open_porch_sf = st.number_input("Open Porch Area (sq ft)", 0, 500, 40, step=10)
    enclosed_porch = st.number_input("Enclosed Porch Area (sq ft)", 0, 500, 0, step=10)

# ---------------------------------------------------------
# 5. Prediction Execution Flow
# ---------------------------------------------------------
# Build dictionary using base defaults and user overrides
raw_data = get_default_row()
raw_data.update({
    'Neighborhood': neighborhood,
    'OverallQual': overall_qual,
    'YearBuilt': year_built,
    'YearRemodAdd': year_remod,
    'YrSold': yr_sold,
    'GrLivArea': float(gr_liv_area),
    '1stFlrSF': float(flr_1st_sf),
    '2ndFlrSF': float(flr_2nd_sf),
    'TotRmsAbvGrd': tot_rooms,
    'FullBath': full_bath,
    'HalfBath': half_bath,
    'BsmtFullBath': float(bsmt_full_bath),
    'BsmtHalfBath': float(bsmt_half_bath),
    'Fireplaces': fireplaces,
    'KitchenQual': kitchen_qual,
    'ExterQual': exter_qual,
    'TotalBsmtSF': float(total_bsmt_sf),
    'GarageCars': garage_cars,
    'GarageArea': float(garage_area),
    'WoodDeckSF': float(wood_deck_sf),
    'OpenPorchSF': float(open_porch_sf),
    'EnclosedPorch': float(enclosed_porch)
})

# Create raw pandas DataFrame
input_df = pd.DataFrame([raw_data])

# Apply feature engineering
input_df = apply_feature_engineering(input_df)

# Prediction Button
if st.button("💰 Predict Property Valuation", type="primary", use_container_width=True):
    if model is None:
        st.error("Model is not loaded. Please ensure `models/best_model.joblib` exists.")
    else:
        try:
            # Predict using pipeline
            predicted_price = model.predict(input_df)[0]
            
            # Display Prediction
            st.success("Valuation Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated Market Price", f"${predicted_price:,.2f}")
            col2.metric("Total Living Area", f"{int(input_df['TotalSF'].iloc[0]):,} sq ft")
            col3.metric("House Age at Sale", f"{int(input_df['HouseAge'].iloc[0])} years")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------
# 6. Model Metadata Expander
# ---------------------------------------------------------
if metadata:
    with st.expander("Model Information & Benchmark Stats"):
        st.json(metadata)