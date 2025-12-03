import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Model, Scaler, and Feature Columns ---
# IMPORTANT: Update these paths to where you saved your files!
try:
    model = pickle.load(open('income_predection.sav', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    train_columns = pickle.load(open('train_columns.pkl', 'rb'))
    # Define numeric columns used for scaling based on your notebook
    NUMERIC_COLS = ["education-num", "capital-gain", "capital-loss", "hours-per-week"]
except FileNotFoundError:
    st.error("Model or necessary files not found. Please ensure 'income_predictor.sav', 'scaler.pkl', and 'train_columns.pkl' are in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during file loading: {e}")
    st.stop()


# --- 2. The Core Data Cleaning Function (for a single user input) ---
def clean_single_input(data_dict, fitted_scaler, TRAIN_FEATURE_COLUMNS, numeric_cols_to_scale):
    """
    Cleans and preprocesses a single row of user input data to match the
    format of the training data features.
    """
    # 1. Convert dictionary to DataFrame
    df = pd.DataFrame([data_dict])

    # Convert numeric fields to appropriate type
    numeric_inputs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numeric_inputs:
        # Use pd.to_numeric with 'coerce' to turn non-numeric inputs into NaN
        # We will assume complete input, but this is a good safeguard.
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # 2. Dropping Columns (education and fnlwgt)
    df.drop(columns=['education', 'fnlwgt'], inplace=True, errors='ignore')

    # 3. Feature Engineering (Occupation Grouping)
    low_impact_occupations = [
        'Adm-clerical', 'Machine-op-inspct', 'Farming-fishing',
        'Armed-Forces', 'Handlers-cleaners', 'Other-service',
        'Priv-house-serv'
    ]
    new_category_name = 'Other-Low-Impact-occupation'
    
    # Replace the low-impact categories if present
    # .item() is used to access the single value in the single-row dataframe
    current_occupation = df['occupation'].iloc[0] 
    if current_occupation in low_impact_occupations:
        df['occupation'].iloc[0] = new_category_name


    # 4. Feature Engineering (Age Binarization)
    AGE_THRESHOLD = 42
    df['is_over_40'] = (df['age'] >= AGE_THRESHOLD).astype(int)
    df.drop('age', axis=1, inplace=True)

    # 5. One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 6. Reindexing: Match all columns from the training set, filling missing OHE columns with 0
    df_final = df_encoded.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # 7. Scaling (Apply the fitted scaler to the numeric columns)
    # Ensure scaling is only applied to the numeric features
    df_final[numeric_cols_to_scale] = fitted_scaler.transform(df_final[numeric_cols_to_scale])

    # Return the processed single-row DataFrame
    return df_final


# --- 3. Streamlit UI and Prediction Logic ---

st.title("Income State Predictor")
st.markdown("Enter the demographic and financial details below to predict the income level (>$50K or <=$50K).")

# Layout columns for cleaner input
col1, col2 = st.columns(2)

# Collect user input using standard text inputs
with col1:
    age = st.text_input('Age', '30')
    workclass = st.text_input('Workclass', 'Private')
    fnlwgt = st.text_input('Fnlwgt (ignored in model)', '170000')
    education = st.text_input('Education Level (e.g., Bachelors)', 'Bachelors')
    education_num = st.text_input('Education Number (e.g., 13)', '13')
    marital_status = st.text_input('Marital Status', 'Married-civ-spouse')
    occupation = st.text_input('Occupation', 'Exec-managerial')
    
with col2:
    relationship = st.text_input('Relationship', 'Husband')
    race = st.text_input('Race', 'White')
    sex = st.text_input('Sex', 'Male')
    capital_gain = st.text_input('Capital Gain', '0')
    capital_loss = st.text_input('Capital Loss', '0')
    hours_per_week = st.text_input('Hours per Week', '40')
    native_country = st.text_input('Native Country', 'United-States')


# Prediction Button
if st.button('Predict Income'):
    # 1. Create the input dictionary from the user's data
    user_input = {
        'age': age,
        'workclass': workclass.strip(), # Clean whitespace from categorical data
        'fnlwgt': fnlwgt,
        'education': education.strip(),
        'education-num': education_num,
        'marital-status': marital_status.strip(),
        'occupation': occupation.strip(),
        'relationship': relationship.strip(),
        'race': race.strip(),
        'sex': sex.strip(),
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country.strip()
    }
    
    # 2. Preprocess the input data
    try:
        processed_data = clean_single_input(
            user_input, 
            scaler, 
            train_columns, 
            NUMERIC_COLS
        )

        # 3. Make Prediction
        # Predict probability of >50K (Class 1)
        probability_high_income = model.predict_proba(processed_data)[:, 1][0]
        
        # Use a threshold (0.35 from your notebook)
        THRESHOLD = 0.1
        prediction = 1 if probability_high_income > THRESHOLD else 0
        
        # 4. Display Result
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.success(f"Based on the inputs, the predicted income is **> $50K**.")
        else:
            st.warning(f"Based on the inputs, the predicted income is **<= $50K**.")
            
        st.info(f"Probability of earning >$50K: **{probability_high_income * 100:.2f}%** (Threshold used: {THRESHOLD})")

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs, especially numeric values. Error: {e}")