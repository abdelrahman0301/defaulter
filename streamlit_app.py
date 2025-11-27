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
                    'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'OWN_CAR_AGE', 'FLAG_MOBIL',
                    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
                    'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                    'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                    'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                    'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_2',
                    'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'FLAG_DOCUMENT_2',
                    'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                    'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                    'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                    'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                    'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                    'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'YEARS_BIRTH',
                    'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'YEARS_ID_PUBLISH', 'YEARS_LAST_PHONE_CHANGE',
                    'PREV_SK_ID_PREV_COUNT', 'PREV_SK_ID_CURR_FIRST', 'PREV_AMT_ANNUITY_MEAN',
                    'PREV_AMT_ANNUITY_MEDIAN', 'PREV_AMT_APPLICATION_MEAN', 'PREV_AMT_APPLICATION_MEDIAN',
                    'PREV_AMT_CREDIT_MEAN', 'PREV_AMT_CREDIT_MEDIAN', 'PREV_AMT_GOODS_PRICE_MEAN',
                    'PREV_AMT_GOODS_PRICE_MEDIAN', 'PREV_NAME_CONTRACT_TYPE_<LAMBDA>',
                    'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>', 'PREV_NAME_CASH_LOAN_PURPOSE_<LAMBDA>',
                    'PREV_NAME_CONTRACT_STATUS_<LAMBDA>', 'PREV_NAME_PAYMENT_TYPE_<LAMBDA>',
                    'PREV_CODE_REJECT_REASON_<LAMBDA>', 'PREV_NAME_CLIENT_TYPE_<LAMBDA>',
                    'PREV_NAME_GOODS_CATEGORY_<LAMBDA>', 'PREV_NAME_PORTFOLIO_<LAMBDA>',
                    'PREV_NAME_PRODUCT_TYPE_<LAMBDA>', 'PREV_CHANNEL_TYPE_<LAMBDA>',
                    'PREV_NAME_SELLER_INDUSTRY_<LAMBDA>', 'PREV_NAME_YIELD_GROUP_<LAMBDA>',
                    'PREV_PRODUCT_COMBINATION_<LAMBDA>', 'PREV_NFLAG_LAST_APPL_IN_DAY_MAX',
                    'PREV_NFLAG_INSURED_ON_APPROVAL_MAX', 'PREV_CNT_PAYMENT_MAX',
                    'PREV_HOUR_APPR_PROCESS_START_MEAN', 'PREV_SELLERPLACE_AREA_MEAN',
                    'PREV_YEARS_DECISION_MEAN', 'PREV_YEARS_FIRST_DRAWING_MEAN',
                    'PREV_YEARS_FIRST_DUE_MEAN', 'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN',
                    'PREV_YEARS_LAST_DUE_MEAN', 'PREV_YEARS_TERMINATION_MEAN'
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
            'CNT_CHILDREN': user_inputs['cnt_children'],
            'CNT_FAM_MEMBERS': user_inputs['cnt_fam_members'],
            'EXT_SOURCE_2': user_inputs['ext_source_2'],
            'EXT_SOURCE_3': user_inputs['ext_source_3'],
            'REGION_RATING_CLIENT': user_inputs['region_rating'],
            'OBS_30_CNT_SOCIAL_CIRCLE': user_inputs['obs_30_cnt'],
            'DEF_30_CNT_SOCIAL_CIRCLE': user_inputs['def_30_cnt'],
            'AMT_REQ_CREDIT_BUREAU_YEAR': user_inputs['amt_req_year'],
            'NAME_CONTRACT_TYPE': user_inputs['contract_type'],
            'CODE_GENDER': user_inputs['gender'],
            'NAME_EDUCATION_TYPE': user_inputs['education'],
        })
        
        return model_inputs
    
    def get_default_inputs(self):
        defaults = {}
        
        defaults['SK_ID_CURR'] = 300000
        defaults['CNT_CHILDREN'] = 0
        defaults['AMT_INCOME_TOTAL'] = 150000
        defaults['AMT_CREDIT'] = 500000
        defaults['AMT_ANNUITY'] = 25000
        defaults['AMT_GOODS_PRICE'] = 450000
        defaults['NAME_CONTRACT_TYPE'] = 'Cash loans'
        defaults['CODE_GENDER'] = 'F'
        defaults['FLAG_OWN_CAR'] = 'N'
        defaults['FLAG_OWN_REALTY'] = 'Y'
        
        defaults['NAME_TYPE_SUITE'] = 'Unaccompanied'
        defaults['NAME_INCOME_TYPE'] = 'Working'
        defaults['NAME_EDUCATION_TYPE'] = 'Secondary / secondary special'
        defaults['NAME_FAMILY_STATUS'] = 'Married'
        defaults['NAME_HOUSING_TYPE'] = 'House / apartment'
        defaults['REGION_POPULATION_RELATIVE'] = 0.02
        defaults['OWN_CAR_AGE'] = 0
        defaults['CNT_FAM_MEMBERS'] = 2
        
        defaults['FLAG_MOBIL'] = 1
        defaults['FLAG_EMP_PHONE'] = 1
        defaults['FLAG_WORK_PHONE'] = 0
        defaults['FLAG_CONT_MOBILE'] = 1
        defaults['FLAG_PHONE'] = 1
        defaults['FLAG_EMAIL'] = 0
        
        defaults['OCCUPATION_TYPE'] = 'Laborers'
        defaults['REGION_RATING_CLIENT'] = 2
        defaults['REGION_RATING_CLIENT_W_CITY'] = 2
        
        defaults['WEEKDAY_APPR_PROCESS_START'] = 'WEDNESDAY'
        defaults['HOUR_APPR_PROCESS_START'] = 10
        
        defaults['REG_REGION_NOT_LIVE_REGION'] = 0
        defaults['REG_REGION_NOT_WORK_REGION'] = 0
        defaults['LIVE_REGION_NOT_WORK_REGION'] = 0
        defaults['REG_CITY_NOT_LIVE_CITY'] = 0
        defaults['REG_CITY_NOT_WORK_CITY'] = 0
        defaults['LIVE_CITY_NOT_WORK_CITY'] = 0
        
        defaults['ORGANIZATION_TYPE'] = 'Business Entity Type 3'
        
        defaults['EXT_SOURCE_2'] = 0.5
        defaults['EXT_SOURCE_3'] = 0.5
        
        defaults['OBS_30_CNT_SOCIAL_CIRCLE'] = 2
        defaults['DEF_30_CNT_SOCIAL_CIRCLE'] = 0
        defaults['OBS_60_CNT_SOCIAL_CIRCLE'] = 2
        defaults['DEF_60_CNT_SOCIAL_CIRCLE'] = 0
        
        doc_flags = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
                    'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
                    'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                    'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
                    'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
        for doc in doc_flags:
            defaults[doc] = 0
        
        defaults['AMT_REQ_CREDIT_BUREAU_HOUR'] = 0
        defaults['AMT_REQ_CREDIT_BUREAU_DAY'] = 0
        defaults['AMT_REQ_CREDIT_BUREAU_WEEK'] = 0
        defaults['AMT_REQ_CREDIT_BUREAU_MON'] = 1
        defaults['AMT_REQ_CREDIT_BUREAU_QRT'] = 0
        defaults['AMT_REQ_CREDIT_BUREAU_YEAR'] = 1
        
        defaults['YEARS_BIRTH'] = -40  
        defaults['YEARS_EMPLOYED'] = -5  
        defaults['YEARS_REGISTRATION'] = -10
        defaults['YEARS_ID_PUBLISH'] = -3
        defaults['YEARS_LAST_PHONE_CHANGE'] = -2
        
        defaults['PREV_SK_ID_PREV_COUNT'] = 1
        defaults['PREV_SK_ID_CURR_FIRST'] = 299999
        defaults['PREV_AMT_ANNUITY_MEAN'] = 20000
        defaults['PREV_AMT_ANNUITY_MEDIAN'] = 20000
        defaults['PREV_AMT_APPLICATION_MEAN'] = 400000
        defaults['PREV_AMT_APPLICATION_MEDIAN'] = 400000
        defaults['PREV_AMT_CREDIT_MEAN'] = 400000
        defaults['PREV_AMT_CREDIT_MEDIAN'] = 400000
        defaults['PREV_AMT_GOODS_PRICE_MEAN'] = 350000
        defaults['PREV_AMT_GOODS_PRICE_MEDIAN'] = 350000
        
        lambda_features = [
            'PREV_NAME_CONTRACT_TYPE_<LAMBDA>', 'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>',
            'PREV_NAME_CASH_LOAN_PURPOSE_<LAMBDA>', 'PREV_NAME_CONTRACT_STATUS_<LAMBDA>',
            'PREV_NAME_PAYMENT_TYPE_<LAMBDA>', 'PREV_CODE_REJECT_REASON_<LAMBDA>',
            'PREV_NAME_CLIENT_TYPE_<LAMBDA>', 'PREV_NAME_GOODS_CATEGORY_<LAMBDA>',
            'PREV_NAME_PORTFOLIO_<LAMBDA>', 'PREV_NAME_PRODUCT_TYPE_<LAMBDA>',
            'PREV_CHANNEL_TYPE_<LAMBDA>', 'PREV_NAME_SELLER_INDUSTRY_<LAMBDA>',
            'PREV_NAME_YIELD_GROUP_<LAMBDA>', 'PREV_PRODUCT_COMBINATION_<LAMBDA>'
        ]
        for feat in lambda_features:
            defaults[feat] = 0
        
        defaults['PREV_NFLAG_LAST_APPL_IN_DAY_MAX'] = 0
        defaults['PREV_NFLAG_INSURED_ON_APPROVAL_MAX'] = 0
        defaults['PREV_CNT_PAYMENT_MAX'] = 36
        
        defaults['PREV_HOUR_APPR_PROCESS_START_MEAN'] = 10
        defaults['PREV_SELLERPLACE_AREA_MEAN'] = 10
        defaults['PREV_YEARS_DECISION_MEAN'] = -1
        defaults['PREV_YEARS_FIRST_DRAWING_MEAN'] = -1
        defaults['PREV_YEARS_FIRST_DUE_MEAN'] = -1
        defaults['PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN'] = -1
        defaults['PREV_YEARS_LAST_DUE_MEAN'] = -1
        defaults['PREV_YEARS_TERMINATION_MEAN'] = -1
        
        return defaults
    
    def predict(self, input_data):
        """Make prediction using the actual LightGBM model"""
        if self.model is None:
            st.error("Model not loaded. Cannot make prediction.")
            return 0.5, False
        
        try:
            input_df = pd.DataFrame([input_data])
            
            missing_features = set(self.features) - set(input_df.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                return 0.5, False
            
            input_df = input_df[self.features]
            
            default_prob = self.model.predict_proba(input_df)[0][1]
            
            return default_prob, default_prob > 0.5
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return 0.5, False

def main():
    st.set_page_config(page_title="Credit Default Risk Prediction", page_icon="💵", layout="wide")
    
    st.title("💵 Credit Default Risk Prediction")
    st.markdown("Predict the likelihood of loan default based on applicant data")
    
    predictor = LoanDefaultPredictor()
    
    if predictor.model is None:
        st.warning("Running in demo mode with sample data")
    
    with st.form("loan_application"):
        st.header("Applicant Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Information")
            income_total = st.number_input("Total Income (AMT_INCOME_TOTAL)", min_value=0, value=150000, step=1000)
            credit_amt = st.number_input("Credit Amount (AMT_CREDIT)", min_value=0, value=500000, step=1000)
            annuity_amt = st.number_input("Annuity Amount (AMT_ANNUITY)", min_value=0, value=25000, step=100)
            goods_price = st.number_input("Goods Price (AMT_GOODS_PRICE)", min_value=0, value=450000, step=1000)
            
        with col2:
            st.subheader("Personal Details")
            cnt_children = st.number_input("Number of Children (CNT_CHILDREN)", min_value=0, value=0)
            cnt_fam_members = st.number_input("Family Members (CNT_FAM_MEMBERS)", min_value=1, value=2)
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=40)
            employment_length = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
            
        with col3:
            st.subheader("External Scores")
            ext_source_2 = st.slider("External Source 2 Score (EXT_SOURCE_2)", 0.0, 1.0, 0.5, 0.01)
            ext_source_3 = st.slider("External Source 3 Score (EXT_SOURCE_3)", 0.0, 1.0, 0.5, 0.01)
            region_rating = st.selectbox("Region Rating (REGION_RATING_CLIENT)", [1, 2, 3], index=1)
            
        st.subheader("Additional Information")
        col4, col5 = st.columns(2)
        
        with col4:
            obs_30_cnt = st.number_input("Observable 30 Days Social Circle (OBS_30_CNT_SOCIAL_CIRCLE)", min_value=0, value=2)
            def_30_cnt = st.number_input("Default 30 Days Social Circle (DEF_30_CNT_SOCIAL_CIRCLE)", min_value=0, value=0)
            amt_req_year = st.number_input("Credit Bureau Requests Year (AMT_REQ_CREDIT_BUREAU_YEAR)", min_value=0, value=1)
            
        with col5:
            contract_type = st.selectbox("Contract Type (NAME_CONTRACT_TYPE)", ["Cash loans", "Revolving loans"])
            gender = st.selectbox("Gender (CODE_GENDER)", ["M", "F"])
            education = st.selectbox("Education (NAME_EDUCATION_TYPE)", [
                "Secondary / secondary special", "Higher education", 
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])
        
        submitted = st.form_submit_button("Predict Default Risk")
    
    if submitted:
        user_inputs = {
            'income_total': income_total,
            'credit_amt': credit_amt,
            'annuity_amt': annuity_amt,
            'goods_price': goods_price,
            'cnt_children': cnt_children,
            'cnt_fam_members': cnt_fam_members,
            'age': age,
            'employment_length': employment_length,
            'ext_source_2': ext_source_2,
            'ext_source_3': ext_source_3,
            'region_rating': region_rating,
            'obs_30_cnt': obs_30_cnt,
            'def_30_cnt': def_30_cnt,
            'amt_req_year': amt_req_year,
            'contract_type': contract_type,
            'gender': gender,
            'education': education,
        }
        
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
            status = "High Risk ❌" if will_default else "Low Risk ✅"
            st.metric(label="Risk Assessment", value=status)
        
        st.subheader("Risk Interpretation")
        if default_prob < 0.3:
            st.success("**LOW RISK**: This applicant shows strong creditworthiness with low probability of default.")
        elif default_prob < 0.6:
            st.warning("**MEDIUM RISK**: This applicant has moderate risk factors. Additional review recommended.")
        else:
            st.error("**HIGH RISK**: This applicant shows significant risk factors for default.")
        
        with st.expander("Technical Details"):
            st.write("Model received these key values:")
            st.write(f"- Age (YEARS_BIRTH): {input_data['YEARS_BIRTH']} years")
            st.write(f"- Employment (YEARS_EMPLOYED): {input_data['YEARS_EMPLOYED']} years")
            st.write(f"- External Score 2: {ext_source_2:.3f}")
            st.write(f"- External Score 3: {ext_source_3:.3f}")

if __name__ == "__main__":
    main()
