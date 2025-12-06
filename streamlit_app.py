import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Defaulter Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('LoanDefaulter_LightGBM.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'LoanDefaulter_LightGBM.pkl' not found.")
    st.stop()

st.title("Loan Defaulter Prediction App")

with st.form("prediction_form"):
    st.header("Applicant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal & ID")
        sk_id_curr = st.number_input("SK_ID_CURR", value=249980)
        years_birth = st.number_input("YEARS_BIRTH (Age)", value=33.2988)
        years_employed = st.number_input("YEARS_EMPLOYED", value=3.761119)
        years_registration = st.number_input("YEARS_REGISTRATION", value=1.138844)
        years_id_publish = st.number_input("YEARS_ID_PUBLISH", value=9.932764)
        cnt_fam_members = st.number_input("CNT_FAM_MEMBERS", value=3.836494)
        occupation_type = st.text_input("OCCUPATION_TYPE", value="Laborers")
        organization_type = st.text_input("ORGANIZATION_TYPE", value="Construction")
        region_pop_relative = st.number_input("REGION_POPULATION_RELATIVE", value=0.018025, format="%.6f")

    with col2:
        st.subheader("Financial & External")
        amt_income_total = st.number_input("AMT_INCOME_TOTAL", value=157500.0)
        amt_credit = st.number_input("AMT_CREDIT", value=313633.648777)
        amt_annuity = st.number_input("AMT_ANNUITY", value=28307.571623)
        amt_goods_price = st.number_input("AMT_GOODS_PRICE", value=287452.596304)
        ext_source_2 = st.number_input("EXT_SOURCE_2", value=0.273767, format="%.6f")
        ext_source_3 = st.number_input("EXT_SOURCE_3", value=0.552359, format="%.6f")
        own_car_age = st.number_input("OWN_CAR_AGE", value=0.0)
        years_last_phone_change = st.number_input("YEARS_LAST_PHONE_CHANGE", value=1.889443)
        amt_req_credit_bureau_year = st.number_input("AMT_REQ_CREDIT_BUREAU_YEAR", value=1.0)

    with col3:
        st.subheader("Process & Social")
        weekday_appr = st.text_input("WEEKDAY_APPR_PROCESS_START", value="SATURDAY")
        hour_appr = st.number_input("HOUR_APPR_PROCESS_START", value=11)
        obs_30 = st.number_input("OBS_30_CNT_SOCIAL_CIRCLE", value=0.775671)
        obs_60 = st.number_input("OBS_60_CNT_SOCIAL_CIRCLE", value=0.387835)

    st.header("Previous Application History (Aggregated)")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        prev_amt_annuity_mean = st.number_input("PREV_AMT_ANNUITY_MEAN", value=26251.311897)
        prev_amt_annuity_median = st.number_input("PREV_AMT_ANNUITY_MEDIAN", value=22013.063837)
        prev_amt_application_mean = st.number_input("PREV_AMT_APPLICATION_MEAN", value=179119.732163)
        prev_amt_application_median = st.number_input("PREV_AMT_APPLICATION_MEDIAN", value=79056.996427)
        prev_amt_credit_mean = st.number_input("PREV_AMT_CREDIT_MEAN", value=347259.459367)
        prev_amt_credit_median = st.number_input("PREV_AMT_CREDIT_MEDIAN", value=438168.884966)
        prev_amt_goods_price_mean = st.number_input("PREV_AMT_GOODS_PRICE_MEAN", value=219756.583087)
        prev_amt_goods_price_median = st.number_input("PREV_AMT_GOODS_PRICE_MEDIAN", value=152713.512619)
        prev_cnt_payment_max = st.number_input("PREV_CNT_PAYMENT_MAX", value=16.654026)
        prev_sellerplace_area_mean = st.number_input("PREV_SELLERPLACE_AREA_MEAN", value=2957.85991)
        prev_sk_id_prev_count = st.number_input("PREV_SK_ID_PREV_COUNT", value=3.387835)
        prev_sk_id_curr_first = st.number_input("PREV_SK_ID_CURR_FIRST", value=249980.660073)

    with exp_col2:
        prev_years_decision_mean = st.number_input("PREV_YEARS_DECISION_MEAN", value=2.535485)
        prev_years_first_due_mean = st.number_input("PREV_YEARS_FIRST_DUE_MEAN", value=100.089508)
        prev_years_last_due_mean = st.number_input("PREV_YEARS_LAST_DUE_MEAN", value=302.413335)
        prev_years_last_due_1st_version_mean = st.number_input("PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN", value=302.27485)
        prev_years_first_drawing_mean = st.number_input("PREV_YEARS_FIRST_DRAWING_MEAN", value=796.84389)
        prev_years_termination_mean = st.number_input("PREV_YEARS_TERMINATION_MEAN", value=302.401239)
        prev_hour_appr_mean = st.number_input("PREV_HOUR_APPR_PROCESS_START_MEAN", value=9.959452)
        prev_nflag_insured_max = st.number_input("PREV_NFLAG_INSURED_ON_APPROVAL_MAX", value=1.0)
        
        prev_prod_comb = st.text_input("PREV_PRODUCT_COMBINATION_<LAMBDA>", value="Card X-Sell")
        prev_goods_cat = st.text_input("PREV_NAME_GOODS_CATEGORY_<LAMBDA>", value="XNA")
        prev_weekday_lambda = st.text_input("PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>", value="MONDAY")

    submit_button = st.form_submit_button("Predict Status")

if submit_button:
    input_data = {
        'ORGANIZATION_TYPE': organization_type,
        'EXT_SOURCE_3': ext_source_3,
        'EXT_SOURCE_2': ext_source_2,
        'YEARS_ID_PUBLISH': years_id_publish,
        'YEARS_EMPLOYED': years_employed,
        'YEARS_REGISTRATION': years_registration,
        'YEARS_BIRTH': years_birth,
        'AMT_ANNUITY': amt_annuity,
        'SK_ID_CURR': sk_id_curr,
        'REGION_POPULATION_RELATIVE': region_pop_relative,
        'YEARS_LAST_PHONE_CHANGE': years_last_phone_change,
        'PREV_SELLERPLACE_AREA_MEAN': prev_sellerplace_area_mean,
        'PREV_YEARS_DECISION_MEAN': prev_years_decision_mean,
        'AMT_CREDIT': amt_credit,
        'PREV_HOUR_APPR_PROCESS_START_MEAN': prev_hour_appr_mean,
        'PREV_YEARS_FIRST_DUE_MEAN': prev_years_first_due_mean,
        'PREV_CNT_PAYMENT_MAX': prev_cnt_payment_max,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_GOODS_PRICE': amt_goods_price,
        'PREV_AMT_ANNUITY_MEAN': prev_amt_annuity_mean,
        'PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN': prev_years_last_due_1st_version_mean,
        'PREV_AMT_ANNUITY_MEDIAN': prev_amt_annuity_median,
        'PREV_SK_ID_PREV_COUNT': prev_sk_id_prev_count,
        'PREV_YEARS_TERMINATION_MEAN': prev_years_termination_mean,
        'PREV_AMT_CREDIT_MEDIAN': prev_amt_credit_median,
        'AMT_REQ_CREDIT_BUREAU_YEAR': amt_req_credit_bureau_year,
        'PREV_YEARS_LAST_DUE_MEAN': prev_years_last_due_mean,
        'OCCUPATION_TYPE': occupation_type,
        'PREV_AMT_APPLICATION_MEDIAN': prev_amt_application_median,
        'PREV_PRODUCT_COMBINATION_<LAMBDA>': prev_prod_comb,
        'PREV_AMT_CREDIT_MEAN': prev_amt_credit_mean,
        'CNT_FAM_MEMBERS': cnt_fam_members,
        'OBS_30_CNT_SOCIAL_CIRCLE': obs_30,
        'OWN_CAR_AGE': own_car_age,
        'PREV_AMT_APPLICATION_MEAN': prev_amt_application_mean,
        'PREV_AMT_GOODS_PRICE_MEDIAN': prev_amt_goods_price_median,
        'HOUR_APPR_PROCESS_START': hour_appr,
        'PREV_AMT_GOODS_PRICE_MEAN': prev_amt_goods_price_mean,
        'PREV_YEARS_FIRST_DRAWING_MEAN': prev_years_first_drawing_mean,
        'PREV_SK_ID_CURR_FIRST': prev_sk_id_curr_first,
        'OBS_60_CNT_SOCIAL_CIRCLE': obs_60,
        'PREV_NAME_GOODS_CATEGORY_<LAMBDA>': prev_goods_cat,
        'PREV_NFLAG_INSURED_ON_APPROVAL_MAX': prev_nflag_insured_max,
        'WEEKDAY_APPR_PROCESS_START': weekday_appr,
        'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>': prev_weekday_lambda
    }

    df = pd.DataFrame([input_data])

    categorical_cols = [
        'ORGANIZATION_TYPE', 
        'OCCUPATION_TYPE', 
        'PREV_PRODUCT_COMBINATION_<LAMBDA>',
        'PREV_NAME_GOODS_CATEGORY_<LAMBDA>',
        'WEEKDAY_APPR_PROCESS_START',
        'PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>'
    ]

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    try:
        raw_prediction = model.predict(df, raw_score=True)[0]
        
        probability = 1 / (1 + np.exp(-raw_prediction))
        
        prediction_class = 1 if probability > 0.5 else 0

        st.subheader("Prediction Results")
        
        if prediction_class == 1:
            st.error(f"Risk Level: High Risk")
        else:
            st.success(f"Risk Level: Low Risk")
            
        st.write(f"**Probability:** {probability:.2%}")
        st.write(f"**Raw Score:** {raw_prediction:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
