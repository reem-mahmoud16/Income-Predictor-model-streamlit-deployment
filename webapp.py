import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Model, Scaler, and Feature Columns ---
# IMPORTANT: Ensure all four files are present in your GitHub repository!
try:
    # Loading model and artifacts
    model = pickle.load(open('income_predection.sav', 'rb')) 
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    train_columns = pickle.load(open('train_columns.pkl', 'rb'))
    # MANDATORY: Load the modes dictionary to handle missing data and maintain pipeline consistency
    imputation_modes = pickle.load(open('imputation_modes.pkl', 'rb')) 

    NUMERIC_COLS = ["education-num", "capital-gain", "capital-loss", "hours-per-week"]
    
except FileNotFoundError as e:
    st.error(f"Missing File Error: {e}. Please ensure all 4 files (.sav, .pkl) are available and uploaded to GitHub.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during file loading: {e}")
    st.stop()


# --- 2. The Core Data Cleaning Function (FIXED FOR STRING CONSISTENCY AND ROBUSTNESS) ---
def clean_single_input(data_dict, fitted_scaler, TRAIN_FEATURE_COLUMNS, numeric_cols_to_scale, modes):
    """
    Cleans and preprocesses a single row of user input data, enforcing the
    exact transformations and consistency used during model training.
    
    This is the core function where mismatches (like the row 3 error) occur.
    """
    df = pd.DataFrame([data_dict])
    
    # 1. Standardize and Impute Categorical Features
    CATEGORICAL_COLS = df.select_dtypes(include=['object']).columns

    for col in CATEGORICAL_COLS:
        # **CRITICAL FIX:** Strip whitespace AND enforce Title Case for perfect OHE match with training data
        # This resolves issues like " Private" (space) or "private" (lowercase)
        df[col] = df[col].astype(str).str.strip().str.title().replace('?', np.nan)
        
        # Imputation (still needed for a robust app)
        if col in modes:
             # Ensure the mode itself is also consistently cased/stripped
             mode_value = modes[col].strip().title() if isinstance(modes[col], str) else modes[col]
             df[col].fillna(mode_value, inplace=True)


    # 2. Convert Numeric Fields to appropriate type (and clean up NaNs)
    numeric_inputs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numeric_inputs:
        # Coerce non-numeric inputs (like empty strings) to NaN, then fill NaN with 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 


    # 3. Dropping Columns (education and fnlwgt)
    df.drop(columns=['education', 'fnlwgt'], inplace=True, errors='ignore')

    # 4. Feature Engineering (Occupation Grouping)
    low_impact_occupations = [
        'Adm-clerical', 'Machine-op-inspct', 'Farming-fishing',
        'Armed-Forces', 'Handlers-cleaners', 'Other-service',
        'Priv-house-serv'
    ]
    new_category_name = 'Other-Low-Impact-occupation'
    
    # Use vectorized .replace() for robust grouping. Needs Title Case input.
    df['occupation'] = df['occupation'].replace(
        [o.title() for o in low_impact_occupations], 
        new_category_name
    )


    # 5. Feature Engineering (Age Binarization)
    AGE_THRESHOLD = 42
    df['is_over_40'] = (df['age'] >= AGE_THRESHOLD).astype(int)
    df.drop('age', axis=1, inplace=True)

    # 6. One-Hot Encoding
    categorical_cols_for_ohe = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols_for_ohe, drop_first=True)

    # 7. Reindexing: Match all columns from the training set, filling missing OHE columns with 0
    # This is vital: it ensures the model always receives the expected column order/count.
    df_final = df_encoded.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # 8. Scaling (Apply the fitted scaler to the numeric columns)
    df_final[numeric_cols_to_scale] = fitted_scaler.transform(df_final[numeric_cols_to_scale])

    return df_final


# --- 3. Streamlit UI and Prediction Logic ---

st.title("Income State Predictor")
st.markdown("Enter the demographic and financial details below to predict the income level (>$50K or <=$50K).")

# Layout columns for cleaner input
col1, col2 = st.columns(2)

# Collect user input using standard text inputs
with col1:
    # Default inputs set to likely trigger a high income prediction for easy testing
    age = st.text_input('Age', '55') 
    workclass = st.text_input('Workclass', 'Private')
    fnlwgt = st.text_input('Fnlwgt (ignored in model)', '170000')
    education = st.text_input('Education Level (e.g., Bachelors)', 'Masters')
    education_num = st.text_input('Education Number (e.g., 13)', '14')
    marital_status = st.text_input('Marital Status', 'Married-civ-spouse')
    occupation = st.text_input('Occupation', 'Exec-managerial')
    
with col2:
    relationship = st.text_input('Relationship', 'Husband')
    race = st.text_input('Race', 'White')
    sex = st.text_input('Sex', 'Male')
    capital_gain = st.text_input('Capital Gain', '0')
    capital_loss = st.text_input('Capital Loss', '0')
    hours_per_week = st.text_input('Hours per Week', '60')
    native_country = st.text_input('Native Country', 'United-States')


# Prediction Button
if st.button('Predict Income'):
    # 1. Create the input dictionary from the user's data
    user_input = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    
    # 2. Preprocess the input data
    try:
        processed_data = clean_single_input(
            user_input, 
            scaler, 
            train_columns, 
            NUMERIC_COLS,
            imputation_modes
        )

        # 3. Make Prediction
        probability_high_income = model.predict_proba(processed_data)[:, 1][0]
        
        THRESHOLD = 0.35
        prediction = 1 if probability_high_income > THRESHOLD else 0
        
        # 4. Display Result
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.success(f"Based on the inputs, the predicted income is **> $50K**.")
        else:
            st.warning(f"Based on the inputs, the predicted income is **<= $50K**.")
            
        st.info(f"Probability of earning >$50K: **{probability_high_income * 100:.2f}%** (Threshold used: {THRESHOLD})")

    except Exception as e:
        st.error(f"An error occurred during prediction. Error details: {e}")