import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class LoanDefaultPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load('LoanDefaulter_LightGBM.pkl')
            
            if hasattr(self.model, 'feature_name_'):
                self.features = self.model.feature_name_
            else:
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
            
            st.success("✅ Credit Default Risk Model loaded successfully!")
            
        except FileNotFoundError:
            st.error("❌ Model file 'LoanDefaulter_LightGBM.pkl' not found.")
            self.model = None
    
    def convert_to_model_format(self, user_inputs):
        model_inputs = self.get_default_inputs()
        
        model_inputs['YEARS_BIRTH'] = -user_inputs['age']
        model_inputs['YEARS_EMPLOYED'] = -user_inputs['employment_length']
        
        model_inputs.update({
            'AMT_INCOME_TOTAL': user_inputs['income_total'],
            'AMT_CREDIT': user_inputs['credit_amt'],
            'AMT_ANNUITY': user_inputs['annuity_amt'],
            'AMT_GOODS_PRICE': user_inputs['goods_price'],
            'CNT_FAM_MEMBERS': user_inputs['cnt_fam_members'],
            'EXT_SOURCE_2': user_inputs['ext_source_2'],
            'EXT_SOURCE_3': user_inputs['ext_source_3'],
            'OBS_30_CNT_SOCIAL_CIRCLE': user_inputs['obs_30_cnt'],
            'AMT_REQ_CREDIT_BUREAU_YEAR': user_inputs['amt_req_year'],
            'OWN_CAR_AGE': user_inputs['own_car_age'],
            'HOUR_APPR_PROCESS_START': user_inputs['hour_appr_process_start'],
            'OBS_60_CNT_SOCIAL_CIRCLE': user_inputs['obs_60_cnt'],
        })
        
        return model_inputs
    
    def get_default_inputs(self):
        defaults = {}
        
        defaults['ORGANIZATION_TYPE'] = 'Business Entity Type 3'
        defaults['EXT_SOURCE_3'] = 0.5
        defaults['EXT_SOURCE_2'] = 0.5
        defaults['YEARS_ID_PUBLISH'] = -3
        defaults['YEARS_EMPLOYED'] = -5
        defaults['YEARS_REGISTRATION'] = -10
        defaults['YEARS_BIRTH'] = -40
        defaults['AMT_ANNUITY'] = 25000
        defaults['SK_ID_CURR'] = 300000
        defaults['REGION_POPULATION_RELATIVE'] = 0.02
        defaults['YEARS_LAST_PHONE_CHANGE'] = -2
        defaults['PREV_SELLERPLACE_AREA_MEAN'] = 10
        defaults['PREV_YEARS_DECISION_MEAN'] = -1
        defaults['AMT_CREDIT'] = 500000
        defaults['PREV_HOUR_APPR_PROCESS_START_MEAN'] = 10
        defaults['PREV_YEARS_FIRST_DUE_MEAN'] = -1
        defaults['PREV_CNT_PAYMENT_MAX'] = 36
        defaults['AMT_INCOME_TOTAL'] = 150000
        defaults['AMT_GOODS_PRICE'] = 450000
        defaults['PREV_AMT_ANNUITY_MEAN'] = 20000
        defaults['PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN'] = -1
        defaults['PREV_AMT_ANNUITY_MEDIAN'] = 20000
        defaults['PREV_SK_ID_PREV_COUNT'] = 1
        defaults['PREV_YEARS_TERMINATION_MEAN'] = -1
        defaults['PREV_AMT_CREDIT_MEDIAN'] = 400000
        defaults['AMT_REQ_CREDIT_BUREAU_YEAR'] = 1
        defaults['PREV_YEARS_LAST_DUE_MEAN'] = -1
        defaults['OCCUPATION_TYPE'] = 'Laborers'
        defaults['PREV_AMT_APPLICATION_MEDIAN'] = 400000
        defaults['PREV_PRODUCT_COMBINATION_<LAMBDA>'] = 0
        defaults['PREV_AMT_CREDIT_MEAN'] = 400000
        defaults['CNT_FAM_MEMBERS'] = 2
        defaults['OBS_30_CNT_SOCIAL_CIRCLE'] = 2
        defaults['OWN_CAR_AGE'] = 0
        defaults['PREV_AMT_APPLICATION_MEAN'] = 400000
        defaults['PREV_AMT_GOODS_PRICE_MEDIAN'] = 350000
        defaults['HOUR_APPR_PROCESS_START'] = 10
        defaults['PREV_AMT_GOODS_PRICE_MEAN'] = 350000
        defaults['PREV_YEARS_FIRST_DRAWING_MEAN'] = -1
        defaults['PREV_SK_ID_CURR_FIRST'] = 299999
        defaults['OBS_60_CNT_SOCIAL_CIRCLE'] = 2
        defaults['PREV_NAME_GOODS_CATEGORY_<LAMBDA>'] = 0
        defaults['PREV_NFLAG_INSURED_ON_APPROVAL_MAX'] = 0
        defaults['WEEKDAY_APPR_PROCESS_START'] = 'WEDNESDAY'
        defaults['PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>'] = 0
        
        return defaults
    
    def predict(self, input_data):
        if self.model is None:
            st.error("Model not loaded. Cannot make prediction.")
            return 0.5, False

        try:
            ordered_data = [input_data.get(f, 0) for f in self.features]
            input_df = pd.DataFrame([ordered_data], columns=self.features)

            string_columns = input_df.select_dtypes(include=['object']).columns

            def encode_as_int(x):
                return abs(hash(str(x))) % 10_000_000  

            for col in string_columns:
                input_df[col] = input_df[col].apply(encode_as_int)

            X = input_df.to_numpy(dtype=float)

            default_prob = self.model.predict(X)[0]
            return default_prob, default_prob > 0.5

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return 0.5, False

def main():
    st.set_page_config(page_title="Credit Default Risk Prediction", page_icon="💵", layout="wide")
    
    st.title("💵 Credit Default Risk Prediction")
    st.markdown("Predict the likelihood of loan default based on applicant data")
    
    predictor = LoanDefaultPredictor()
    
    if predictor.model is None:
        st.warning("Running in demo mode with sample data")
    
    user_inputs = None
    input_data = None
    
    with st.form("loan_application"):
        st.header("Applicant Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Financial Information")
            income_total = st.number_input("Total Income (AMT_INCOME_TOTAL)", min_value=0, value=150000, step=1000)
            credit_amt = st.number_input("Credit Amount (AMT_CREDIT)", min_value=0, value=500000, step=1000)
            annuity_amt = st.number_input("Annuity Amount (AMT_ANNUITY)", min_value=0, value=25000, step=100)
            goods_price = st.number_input("Goods Price (AMT_GOODS_PRICE)", min_value=0, value=450000, step=1000)
            
        with col2:
            st.subheader("Personal Details")
            cnt_fam_members = st.number_input("Family Members (CNT_FAM_MEMBERS)", min_value=1, value=2)
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=40)
            employment_length = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
            own_car_age = st.number_input("Car Age (OWN_CAR_AGE)", min_value=0, value=0)
            
        with col3:
            st.subheader("External & Social")
            ext_source_2 = st.slider("External Source 2 (EXT_SOURCE_2)", 0.0, 1.0, 0.5, 0.01)
            ext_source_3 = st.slider("External Source 3 (EXT_SOURCE_3)", 0.0, 1.0, 0.5, 0.01)
            obs_30_cnt = st.number_input("Social Circle 30 (OBS_30_CNT)", min_value=0, value=2)
            obs_60_cnt = st.number_input("Social Circle 60 (OBS_60_CNT)", min_value=0, value=2)
            amt_req_year = st.number_input("Credit Bureau Reqs/Year", min_value=0, value=1)
            hour_appr_process_start = st.number_input("Appr. Hour", min_value=0, max_value=23, value=10)
        
        submitted = st.form_submit_button("Predict Default Risk")
        
        if submitted:
            user_inputs = {
                'income_total': income_total,
                'credit_amt': credit_amt,
                'annuity_amt': annuity_amt,
                'goods_price': goods_price,
                'cnt_fam_members': cnt_fam_members,
                'age': age,
                'employment_length': employment_length,
                'own_car_age': own_car_age,
                'ext_source_2': ext_source_2,
                'ext_source_3': ext_source_3,
                'obs_30_cnt': obs_30_cnt,
                'obs_60_cnt': obs_60_cnt,
                'amt_req_year': amt_req_year,
                'hour_appr_process_start': hour_appr_process_start
            }
    
    if submitted and user_inputs is not None:
        input_data = predictor.convert_to_model_format(user_inputs)
        
        with st.spinner("Analyzing application..."):
            default_prob, will_default = predictor.predict(input_data)
        
        st.header("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Default Probability", 
                value=f"{default_prob:.1%}",
                delta=f"{(default_prob - 0.5):.1%}" if default_prob > 0.5 else f"{(0.5 - default_prob):.1%}",
                delta_color="inverse"
            )
            
        with col2:
            if default_prob < 0.3:
                status = "Low Risk ✅"
            elif default_prob < 0.6:
                status = "Medium Risk ⚠️"
            else:
                status = "High Risk ❌"
            st.metric(label="Risk Assessment", value=status)
        
        st.subheader("Risk Interpretation")
        if default_prob
