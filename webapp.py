import streamlit as st
import pandas as pd
import pickle
import numpy as np


try:
    model = pickle.load(open('income_predection.sav', 'rb')) 
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    train_columns = pickle.load(open('train_columns.pkl', 'rb'))
    imputation_modes = pickle.load(open('imputation_modes.pkl', 'rb')) 

    NUMERIC_COLS = ["education-num", "capital-gain", "capital-loss", "hours-per-week"]
    
except FileNotFoundError as e:
    st.error(f"Missing File Error: {e}. Please ensure all 4 files (.sav, .pkl) are available and uploaded to GitHub.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during file loading: {e}")
    st.stop()


# --- The Core Data Cleaning Function
def clean_single_input(data_dict, fitted_scaler, TRAIN_FEATURE_COLUMNS, numeric_cols_to_scale, modes):
    """
    Cleans and preprocesses a single row of user input data, enforcing the
    exact transformations and consistency used during model training.
    
    This is the core function where mismatches (like the row 3 error) occur.
    """
    df = pd.DataFrame([data_dict])
    
    CATEGORICAL_COLS = df.select_dtypes(include=['object']).columns

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.title().replace('?', np.nan)
        
        if col in modes:
             mode_value = modes[col].strip().title() if isinstance(modes[col], str) else modes[col]
             df[col].fillna(mode_value, inplace=True)


    numeric_inputs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numeric_inputs:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 


    df.drop(columns=['education', 'fnlwgt'], inplace=True, errors='ignore')

    low_impact_occupations = [
        'Adm-clerical', 'Machine-op-inspct', 'Farming-fishing',
        'Armed-Forces', 'Handlers-cleaners', 'Other-service',
        'Priv-house-serv'
    ]
    new_category_name = 'Other-Low-Impact-occupation'
    
    df['occupation'] = df['occupation'].replace(
        [o.title() for o in low_impact_occupations], 
        new_category_name
    )


    AGE_THRESHOLD = 42
    df['is_over_40'] = (df['age'] >= AGE_THRESHOLD).astype(int)
    df.drop('age', axis=1, inplace=True)

    categorical_cols_for_ohe = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols_for_ohe, drop_first=True)

    df_final = df_encoded.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    df_final[numeric_cols_to_scale] = fitted_scaler.transform(df_final[numeric_cols_to_scale])

    return df_final


# --- Streamlit UI and Prediction Logic ---

st.title("Income State Predictor")
st.markdown("Enter the demographic and financial details below to predict the income level (>$50K or <=$50K).")

col1, col2 = st.columns(2)

with col1:
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
    
    # Preprocess the input data
    try:
        processed_data = clean_single_input(
            user_input, 
            scaler, 
            train_columns, 
            NUMERIC_COLS,
            imputation_modes
        )

        # Make Prediction
        probability_high_income = model.predict_proba(processed_data)[:, 1][0]
        
        THRESHOLD = 0.35
        prediction = 1 if probability_high_income > THRESHOLD else 0
        
        # Display Result
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.success(f"Based on the inputs, the predicted income is **> $50K**.")
        else:
            st.warning(f"Based on the inputs, the predicted income is **<= $50K**.")
            
        st.info(f"Probability of earning >$50K: **{probability_high_income * 100:.2f}%** (Threshold used: {THRESHOLD})")

    except Exception as e:
        st.error(f"An error occurred during prediction. Error details: {e}")