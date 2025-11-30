import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import lightgbm # <-- NEW IMPORT

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

class LoanDefaultPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.cat_mappings = {} 
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load('LoanDefaulter_LightGBM.pkl')
            
            # 1. Load Feature Names
            if hasattr(self.model, 'feature_name_'):
                self.features = self.model.feature_name_
            
            # If the model is the raw Booster, we must rely on its feature names
            elif isinstance(self.model, lightgbm.Booster):
                self.features = self.model.feature_name()
                
            else:
                # Fallback list if feature names aren't in the object
                self.features = [
                    'ORGANIZATION_TYPE', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'YEARS_ID_PUBLISH', 
                    'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'YEARS_BIRTH', 'AMT_ANNUITY', 
                    'SK_ID_CURR', 'REGION_POPULATION_RELATIVE', 'YEARS_LAST_PHONE_CHANGE', 
                    'PREV_SELLERPLACE_AREA_MEAN', 'PREV_YEARS_DECISION_MEAN', 'AMT_CREDIT', 
                    'PREV_HOUR_APPR_PROCESS_START_MEAN', 'PREV_YEARS_FIRST_DUE_MEAN', 
                    'PREV_CNT_PAYMENT_MAX', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 
                    'PREV_AMT_ANNUITY_MEAN', 'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN', 
                    'PREV_AMT_ANNUITY_MEDIAN', 'PREV_SK_ID_PREV_COUNT', 
                    'PREV_YEARS_TERMINATION_MEAN', 'PREV_AMT_CREDIT_MEDIAN', 
                    'AMT_REQ_CREDIT_BUREAU_YEAR', 'PREV_YEARS_LAST_DUE_MEAN', 
                    'OCCUPATION_TYPE', 'PREV_AMT_APPLICATION_MEDIAN', 
                    'PREV_PRODUCT_COMBINATION_<LAMBDA>', 'PREV_AMT_CREDIT_MEAN', 
                    'CNT_FAM_MEMBERS', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OWN_CAR_AGE', 
                    'PREV_AMT_APPLICATION_MEAN', 'PREV_AMT_GOODS_PRICE_MEDIAN', 
                    'HOUR_APPR_PROCESS_START', 'PREV_AMT_GOODS_PRICE_MEAN', 
                    'PREV_YEARS_FIRST_DRAWING_MEAN', 'PREV_SK_ID_CURR_FIRST', 
                    'OBS_60_CNT_SOCIAL_CIRCLE', 'PREV_NAME_GOODS_CATEGORY_<LAMBDA>', 
                    'PREV_NFLAG_INSURED_ON_APPROVAL_MAX', 'WEEKDAY_APPR_PROCESS_START', 
                    'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>'
                ]
            
            # 2. Extract Category Mappings (Critical Fix for Hashing Issue)
            try:
                # Check if it is the sklearn wrapper (has booster_ attribute)
                if hasattr(self.model, 'booster_'):
                    booster = self.model.booster_
                # Check if it is the raw Booster object itself
                elif isinstance(self.model, lightgbm.Booster):
                    booster = self.model
                else:
                    booster = None

                if booster and hasattr(booster, 'pandas_categorical'):
                    # Access the internal LightGBM booster to find category lists
                    self.cat_mappings = {
                        name: list(cats) 
                        for name, cats in zip(
                            booster.feature_name(), 
                            booster.pandas_categorical
                        )
                    }
            except Exception as e:
                print(f"Warning: Could not extract specific category mappings. {e}")
            
            st.success("✅ Credit Default Risk Model loaded successfully!")
            
        except FileNotFoundError:
            st.error("❌ Model file 'LoanDefaulter_LightGBM.pkl' not found. Please upload it.")
            self.model = None
    
    # We remove this empty function
    # def convert_to_model_format(self, inputs):
    #     return inputs.copy()
    
    def predict(self, input_data):
        if self.model is None:
            st.error("Model not loaded. Cannot make prediction.")
            return 0.5, False

        try:
            # Create a DataFrame with a single row
            input_df = pd.DataFrame([input_data])

            # Ensure all required columns exist (fill missing with 0)
            for f in self.features:
                if f not in input_df.columns:
                    input_df[f] = 0

            # Reorder columns to match the model's expectation exactly
            input_df = input_df[self.features]

            # --- CRITICAL FIX: Categorical Encoding ---
            for col in input_df.columns:
                if input_df[col].dtype == 'object' or col in self.cat_mappings:
                    
                    if col in self.cat_mappings:
                        categories = self.cat_mappings[col]
                        val = input_df.iloc[0][col]
                        
                        try:
                            # Convert string to the integer index (e.g., 'Laborers' -> 4)
                            encoded_val = categories.index(val)
                        except ValueError:
                            # If the user input isn't in the training list, use -1 (unknown)
                            encoded_val = -1
                            
                        input_df[col] = encoded_val
                    else:
                        # Fallback if we don't have the mapping: assume missing/unknown
                        if isinstance(input_df.iloc[0][col], str):
                             input_df[col] = -1 

            # Convert to float numpy array
            X = input_df.to_numpy(dtype=float)

            # --- CRITICAL FIX: Handling Booster vs LGBMClassifier Prediction ---
            if isinstance(self.model, lightgbm.Booster):
                # Raw Booster uses .predict() to return probability scores for binary objective
                prob_scores = self.model.predict(X)
                default_prob = prob_scores[0]
            else:
                # Scikit-learn wrapper uses .predict_proba()
                default_prob = self.model.predict_proba(X)[0][1]
            
            return default_prob, default_prob > 0.5

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return 0.5, False

def main():
    st.set_page_config(page_title="Credit Default Risk Prediction", page_icon="💵", layout="wide")
    
    st.title("💵 Credit Default Risk Prediction")
    st.markdown("Predict the likelihood of loan default. Please fill in all applicant details below.")
    
    predictor = LoanDefaultPredictor()
    
    if predictor.model is None:
        st.warning("Running in demo mode without model file.")
    
    with st.form("loan_application"):
        
        st.header("1. Personal & Employment Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sk_id_curr = st.number_input("ID (SK_ID_CURR)", value=100001)
            # FIX: Training data uses positive float years, do not convert to negative later
            years_birth = st.number_input("Age (Years)", min_value=18.0, value=30.0) 
            cnt_fam_members = st.number_input("Family Members", min_value=1.0, value=1.0)
        with c2:
            years_employed = st.number_input("Years Employed", min_value=0.0, value=2.0)
            years_registration = st.number_input("Years Since Registration", min_value=0.0, value=5.0)
            years_id_publish = st.number_input("Years Since ID Publish", min_value=0.0, value=2.0)
        with c3:
            # Common Home Credit categories
            occupation = st.selectbox("Occupation Type", 
                                      ["Laborers", "Sales staff", "Accountants", "Managers", "Drivers", "Core staff", "High skill tech staff", "Medicine staff"])
            organization = st.selectbox("Organization Type", 
                                        ["Business Entity Type 3", "Self-employed", "Other", "Government", "Medicine"])
            years_last_phone = st.number_input("Years Since Phone Change", min_value=0.0, value=1.0)
        with c4:
            weekday_appr = st.selectbox("Weekday of Application", 
                                        ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY'])
            hour_appr = st.number_input("Hour of Application", 0, 23, 10)

        st.header("2. Financial Information")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            income = st.number_input("Total Income", value=150000.0)
            credit = st.number_input("Credit Amount", value=400000.0)
        with c6:
            annuity = st.number_input("Annuity Amount", value=20000.0)
            goods_price = st.number_input("Goods Price", value=350000.0)
        with c7:
            own_car_age = st.number_input("Car Age (0 if none)", value=0.0)
            region_pop = st.number_input("Region Population Relative", value=0.0188, format="%.4f")
        with c8:
             amt_req_year = st.number_input("Enquiries to Bureau (Year)", value=1.0)

        st.header("3. External Scores & Social")
        c9, c10, c11 = st.columns(3)
        with c9:
            ext2 = st.slider("External Source 2", 0.0, 1.0, 0.5)
            ext3 = st.slider("External Source 3", 0.0, 1.0, 0.5)
        with c10:
            obs_30 = st.number_input("Obs. 30 Social Circle", value=0.0)
            obs_60 = st.number_input("Obs. 60 Social Circle", value=0.0)
            
        st.header("4. History & Aggregated Data")
        st.info("Aggregated statistics from previous loan applications.")
        
        with st.expander("Expand to enter Previous Application History", expanded=True):
            cols_hist = st.columns(4)
            
            with cols_hist[0]:
                st.markdown("**Previous Amounts**")
                prev_cred_mean = st.number_input("Prev Credit Mean", value=100000.0)
                prev_cred_med = st.number_input("Prev Credit Median", value=100000.0)
                prev_app_mean = st.number_input("Prev App Amount Mean", value=100000.0)
                prev_app_med = st.number_input("Prev App Amount Median", value=100000.0)
                prev_ann_mean = st.number_input("Prev Annuity Mean", value=10000.0)
                prev_ann_med = st.number_input("Prev Annuity Median", value=10000.0)
                prev_goods_mean = st.number_input("Prev Goods Mean", value=100000.0)
                prev_goods_med = st.number_input("Prev Goods Median", value=100000.0)

            with cols_hist[1]:
                st.markdown("**Previous Timing (Means)**")
                prev_dec_mean = st.number_input("Years Decision Mean", value=-1.0)
                prev_first_draw = st.number_input("Years First Draw Mean", value=365243.0) 
                prev_first_due = st.number_input("Years First Due Mean", value=-1.0)
                prev_last_due_1st = st.number_input("Years Last Due 1st Ver.", value=-1.0)
                prev_last_due = st.number_input("Years Last Due Mean", value=-1.0)
                prev_term = st.number_input("Years Termination Mean", value=-1.0)

            with cols_hist[2]:
                st.markdown("**Previous Counts & Logic**")
                prev_count = st.number_input("Previous Apps Count", value=1.0)
                prev_cnt_pay_max = st.number_input("Max Count Payments", value=12.0)
                prev_seller_area = st.number_input("Seller Place Area Mean", value=50.0)
                prev_hour_mean = st.number_input("Appr. Hour Mean", value=12.0)
                prev_sk_id_first = st.number_input("First Previous ID", value=100000.0)
                prev_insured = st.number_input("Insured on Approval Max (0/1)", 0.0, 1.0, 0.0)

            with cols_hist[3]:
                st.markdown("**Encoded/Lambda Scores**")
                # FIX: These are STRINGS/CATEGORIES in the model, not numbers.
                # We use selectbox with common values (or XNA if unknown).
                prev_prod_comb = st.selectbox("Prev Prod Combination", 
                                             ["Cash", "POS household with interest", "POS mobile with interest", "Card Street", "XNA"])
                
                prev_goods_cat = st.selectbox("Prev Goods Category", 
                                             ["XNA", "Mobile", "Consumer Electronics", "Computers", "Audio/Video", "Furniture"])
                
                prev_weekday_lambda = st.selectbox("Prev Weekday Process", 
                                                  ['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY'])

        submitted = st.form_submit_button("Predict Default Risk")

    if submitted:
        # Construct dictionary with raw user inputs
        user_data = {
            'ORGANIZATION_TYPE': organization,
            'EXT_SOURCE_3': ext3,
            'EXT_SOURCE_2': ext2,
            'YEARS_ID_PUBLISH': years_id_publish, 
            'YEARS_EMPLOYED': years_employed,     
            'YEARS_REGISTRATION': years_registration, 
            'YEARS_BIRTH': years_birth,         
            'AMT_ANNUITY': annuity,
            'SK_ID_CURR': sk_id_curr,
            'REGION_POPULATION_RELATIVE': region_pop,
            'YEARS_LAST_PHONE_CHANGE': years_last_phone, 
            'PREV_SELLERPLACE_AREA_MEAN': prev_seller_area,
            'PREV_YEARS_DECISION_MEAN': prev_dec_mean,
            'AMT_CREDIT': credit,
            'PREV_HOUR_APPR_PROCESS_START_MEAN': prev_hour_mean,
            'PREV_YEARS_FIRST_DUE_MEAN': prev_first_due,
            'PREV_CNT_PAYMENT_MAX': prev_cnt_pay_max,
            'AMT_INCOME_TOTAL': income,
            'AMT_GOODS_PRICE': goods_price,
            'PREV_AMT_ANNUITY_MEAN': prev_ann_mean,
            'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN': prev_last_due_1st,
            'PREV_AMT_ANNUITY_MEDIAN': prev_ann_med,
            'PREV_SK_ID_PREV_COUNT': prev_count,
            'PREV_YEARS_TERMINATION_MEAN': prev_term,
            'PREV_AMT_CREDIT_MEDIAN': prev_cred_med,
            'AMT_REQ_CREDIT_BUREAU_YEAR': amt_req_year,
            'PREV_YEARS_LAST_DUE_MEAN': prev_last_due,
            'OCCUPATION_TYPE': occupation,
            'PREV_AMT_APPLICATION_MEDIAN': prev_app_med,
            'PREV_PRODUCT_COMBINATION_<LAMBDA>': prev_prod_comb,
            'PREV_AMT_CREDIT_MEAN': prev_cred_mean,
            'CNT_FAM_MEMBERS': cnt_fam_members,
            'OBS_30_CNT_SOCIAL_CIRCLE': obs_30,
            'OWN_CAR_AGE': own_car_age,
            'PREV_AMT_APPLICATION_MEAN': prev_app_mean,
            'PREV_AMT_GOODS_PRICE_MEDIAN': prev_goods_med,
            'HOUR_APPR_PROCESS_START': hour_appr,
            'PREV_AMT_GOODS_PRICE_MEAN': prev_goods_mean,
            'PREV_YEARS_FIRST_DRAWING_MEAN': prev_first_draw,
            'PREV_SK_ID_CURR_FIRST': prev_sk_id_first,
            'OBS_60_CNT_SOCIAL_CIRCLE': obs_60,
            'PREV_NAME_GOODS_CATEGORY_<LAMBDA>': prev_goods_cat,
            'PREV_NFLAG_INSURED_ON_APPROVAL_MAX': prev_insured,
            'WEEKDAY_APPR_PROCESS_START': weekday_appr,
            'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>': prev_weekday_lambda
        }
        
        with st.spinner("Calculating Risk..."):
            prob, is_default = predictor.predict(user_data)
            
        st.divider()
        st.header("Results")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Default Probability", f"{prob:.2%}", delta_color="inverse")
            
        with col_res2:
            if prob < 0.5:
                st.success(f"**Low/Medium Risk** (Probability: {prob:.2%})")
            else:
                st.error(f"**High Risk** (Probability: {prob:.2%})")

if __name__ == "__main__":
    main()
