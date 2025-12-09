import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import google.generativeai as genai


genai.configure(api_key=st.secrets["gemini"]["api_key"])

SYSTEM_INSTRUCTION = (
    "You are an expert system in the field of loans and credit risk. Your primary task is to provide concise, accurate, and helpful answers in based on the conversation context."
    "If you receive a message starting with 'SYSTEM LOG:', read the information and store it in context, but **do not generate a response** to the user for that log message."
    "column SK_ID_CURR means ID of loan in our sample."
    "column YEARS_BIRTH (Age) means Client's age in years at the time of application."
    "column YEARS_EMPLOYED means How many years before the application the person started current employment."
    "column YEARS_REGISTRATION means How many years before the application did client change his registration."
    "column YEARS_ID_PUBLISH means How many years before the application did client change the identity document with which he applied for the loan."
    "column CNT_FAM_MEMBERS means How many family members does client have."
    "column OCCUPATION_TYPE means What kind of occupation does the client have."
    "column ORGANIZATION_TYPE means Type of organization where client works."
    "column REGION_POPULATION_RELATIV means Normalized population of region where client lives (higher number means the client lives in more populated region)."
    "column AMT_INCOME_TOTAL means Income of the client."
    "column AMT_CREDIT means Credit amount of the loan."
    "column AMT_ANNUITY means Loan annuity."
    "column AMT_GOODS_PRICE For consumer loans it is the price of the goods for which the loan is given."
    "column EXT_SOURCE_2 means Normalized score from external data source."
    "column EXT_SOURCE_3 means Normalized score from external data source."
    "column OWN_CAR_AGE means Age of client's car."
    "column YEARS_LAST_PHONE_CHANGE means How many years before application did client change phone."
    "column AMT_REQ_CREDIT_BUREAU_YEAR means Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)."
    "column WEEKDAY_APPR_PROCESS_START means On which day of the week did the client apply for previous application."
    "column HOUR_APPR_PROCESS_START means Approximately at what day hour did the client apply for the previous application."
    "column OBS_30_CNT_SOCIAL_CIRCL means How many observation of client's social surroundings with observable 30 DPD (days past due) default."
    "column OBS_60_CNT_SOCIAL_CIRCLE means How many observation of client's social surroundings with observable 60 DPD (days past due) default."
    "column CNT_PAYMENT means Term of previous credit at application of the previous application."
    "column SELLERPLACE_AREA means Selling area of seller place of the previous application."
    "column YEARS_DECISION means Relative to current application when was the decision about previous application made."
    "column YEARS_FIRST_DUE means Relative to application date of current application when was the first due supposed to be of the previous application."
    "column YEARS_LAST_DUE means Relative to application date of current application when was the last due date of the previous application."
    "column YEARS_LAST_DUE_1ST_VERSION means Relative to application date of current application when was the first due of the previous application."
    "column YEARS_FIRST_DRAWING means Relative to application date of current application when was the first disbursement of the previous application."
    "column YEARS_TERMINATION means Relative to application date of current application when was the expected termination of the previous application."
    "column HOUR_APPR_PROCESS_START means Approximately at what hour did the client apply for the loan."
    "column NFLAG_INSURED_ON_APPROVAL means Did the client requested insurance during the previous application."
    "column PRODUCT_COMBINATION means Detailed product combination of the previous application."
    "column NAME_GOODS_CATEGORY means What kind of goods did the client apply for in the previous application."
    "column WEEKDAY_APPR_PROCESS_START means On which day of the week did the client apply for the loan."
    "If the column has previous in its name, it refers to the previous loan application history of the client (aggregated)."
)

MODEL_NAME = "gemini-2.5-flash"

if "chat_session" not in st.session_state:
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        st.session_state.chat_session = model.start_chat()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")



st.set_page_config(page_title="Loan Defaulter Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('LoanDefaulter_LightGBM.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'LoanDefaulter_LightGBM.pkl' not found.")
    st.stop()

st.title("💵 Loan Defaulter Prediction")

with st.form("prediction_form"):
    st.header("Applicant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal & ID")
        sk_id_curr = st.number_input("SK_ID_CURR", value=403354)
        years_birth = st.number_input("YEARS_BIRTH (Age)", value=62.239562)
        years_employed = st.number_input("YEARS_EMPLOYED", value=5.10883)
        years_registration = st.number_input("YEARS_REGISTRATION", value=24.454483)
        years_id_publish = st.number_input("YEARS_ID_PUBLISH", value=12.755647)
        cnt_fam_members = st.number_input("CNT_FAM_MEMBERS", value=2.0)
        
        occupation_options = ['Laborers', 'Accountants', 'Managers', 'Sales staff', 'Drivers', 'Core staff', 
                              'Medicine staff', 'High skill tech staff', 'Secretaries', 'Waiters/barmen staff', 
                              'Cooking staff', 'Realty agents', 'Cleaning staff', 'Low-skill Laborers', 
                              'Private service staff', 'Security staff', 'HR staff', 'IT staff']
        occupation_type = st.selectbox("OCCUPATION_TYPE", options=occupation_options, index=occupation_options.index("Laborers"))
        
        organization_options = ['Business Entity Type 3', 'Government', 'Other', 'Trade: type 7', 'Business Entity Type 2', 
                                'Security Ministries', 'Self-employed', 'Construction', 'Transport: type 4', 'Trade: type 2', 
                                'Housing', 'Industry: type 3', 'Military', 'Trade: type 3', 'Business Entity Type 1', 
                                'Industry: type 2', 'School', 'Kindergarten', 'Industry: type 9', 'Medicine', 'Emergency', 
                                'Industry: type 11', 'Police', 'Industry: type 5', 'Industry: type 10', 'Postal', 
                                'Industry: type 4', 'Agriculture', 'Bank', 'Industry: type 12', 'University', 
                                'Transport: type 2', 'Services', 'Transport: type 3', 'Industry: type 7', 'Restaurant', 
                                'Telecom', 'Security', 'Mobile', 'Industry: type 1', 'Cleaning', 'Insurance', 'Electricity', 
                                'Religion', 'Advertising', 'Trade: type 1', 'Legal Services', 'Realtor', 'Trade: type 6', 
                                'Culture', 'Hotel', 'Transport: type 1', 'Industry: type 6', 'Industry: type 13', 
                                'Industry: type 8', 'Trade: type 4', 'Trade: type 5']
        organization_type = st.selectbox("ORGANIZATION_TYPE", options=organization_options, index=organization_options.index("Business Entity Type 3"))
        
        region_pop_relative = st.number_input("REGION_POPULATION_RELATIVE", value=0.035792, format="%.6f")

    with col2:
        st.subheader("Financial & External")
        amt_income_total = st.number_input("AMT_INCOME_TOTAL", value=211500.0)
        amt_credit = st.number_input("AMT_CREDIT", value=883863.0)
        amt_annuity = st.number_input("AMT_ANNUITY", value=25330.5)
        amt_goods_price = st.number_input("AMT_GOODS_PRICE", value=737784.0)
        ext_source_2 = st.number_input("EXT_SOURCE_2", value=0.768663, format="%.6f")
        ext_source_3 = st.number_input("EXT_SOURCE_3", value=0.488455, format="%.6f")
        own_car_age = st.number_input("OWN_CAR_AGE", value=4.0)
        years_last_phone_change = st.number_input("YEARS_LAST_PHONE_CHANGE", value=0.0)
        amt_req_credit_bureau_year = st.number_input("AMT_REQ_CREDIT_BUREAU_YEAR", value=2.0)

    with col3:
        st.subheader("Process & Social")
        weekday_options = ['WEDNESDAY', 'MONDAY', 'SUNDAY', 'THURSDAY', 'SATURDAY', 'FRIDAY', 'TUESDAY']
        weekday_appr = st.selectbox("WEEKDAY_APPR_PROCESS_START", options=weekday_options, index=weekday_options.index("SATURDAY"))
        
        hour_appr = st.number_input("HOUR_APPR_PROCESS_START", value=14)
        obs_30 = st.number_input("OBS_30_CNT_SOCIAL_CIRCLE", value=4.0)
        obs_60 = st.number_input("OBS_60_CNT_SOCIAL_CIRCLE", value=4.0)

    st.header("Previous Application History (Aggregated)")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        prev_amt_annuity_mean = st.number_input("PREV_AMT_ANNUITY_MEAN", value=21739.23)
        prev_amt_annuity_median = st.number_input("PREV_AMT_ANNUITY_MEDIAN", value=21739.23)
        prev_amt_application_mean = st.number_input("PREV_AMT_APPLICATION_MEAN", value=0.0)
        prev_amt_application_median = st.number_input("PREV_AMT_APPLICATION_MEDIAN", value=0.0)
        prev_amt_credit_mean = st.number_input("PREV_AMT_CREDIT_MEAN", value=0.0)
        prev_amt_credit_median = st.number_input("PREV_AMT_CREDIT_MEDIAN", value=0.0)
        prev_amt_goods_price_mean = st.number_input("PREV_AMT_GOODS_PRICE_MEAN", value=0.0)
        prev_amt_goods_price_median = st.number_input("PREV_AMT_GOODS_PRICE_MEDIAN", value=0.0)
        prev_cnt_payment_max = st.number_input("PREV_CNT_PAYMENT_MAX", value=24.0)
        prev_sellerplace_area_mean = st.number_input("PREV_SELLERPLACE_AREA_MEAN", value=-1.0)
        prev_sk_id_prev_count = st.number_input("PREV_SK_ID_PREV_COUNT", value=1.0)
        prev_sk_id_curr_first = st.number_input("PREV_SK_ID_CURR_FIRST", value=403354.0)

    with exp_col2:
        prev_years_decision_mean = st.number_input("PREV_YEARS_DECISION_MEAN", value=0.50924)
        prev_years_first_due_mean = st.number_input("PREV_YEARS_FIRST_DUE_MEAN", value=2.220397)
        prev_years_last_due_mean = st.number_input("PREV_YEARS_LAST_DUE_MEAN", value=2.672142)
        prev_years_last_due_1st_version_mean = st.number_input("PREV_YEARS_LAST_DUE_1ST_VERSION_MEAN", value=1.626283)
        prev_years_first_drawing_mean = st.number_input("PREV_YEARS_FIRST_DRAWING_MEAN", value=999.980835)
        prev_years_termination_mean = st.number_input("PREV_YEARS_TERMINATION_MEAN", value=2.688569)
        prev_hour_appr_mean = st.number_input("PREV_HOUR_APPR_PROCESS_START_MEAN", value=13.0)
        prev_nflag_insured_max = st.number_input("PREV_NFLAG_INSURED_ON_APPROVAL_MAX", value=0.0)
        
        prev_prod_comb_options = ['POS other with interest', 'POS mobile without interest', 'POS household with interest', 
                                  'POS industry without interest', 'Cash X-Sell: high', 'POS industry with interest', 'Cash', 
                                  'Cash Street: low', 'POS mobile with interest', 'Cash Street: high', 'Card X-Sell', 
                                  'Cash X-Sell: low', 'Card Street', 'Cash X-Sell: middle', 'POS household without interest', 
                                  'Cash Street: middle', 'POS others without interest']
        prev_prod_comb = st.selectbox("PREV_PRODUCT_COMBINATION_<LAMBDA>", options=prev_prod_comb_options, index=prev_prod_comb_options.index("Cash"))
        
        prev_goods_cat_options = ['Vehicles', 'Mobile', 'Audio/Video', 'Furniture', 'XNA', 'Clothing and Accessories', 
                                  'Computers', 'Consumer Electronics', 'Auto Accessories', 'Medicine', 'Photo / Cinema Equipment', 
                                  'Office Appliances', 'Jewelry', 'Construction Materials', 'Gardening', 'Homewares', 
                                  'Medical Supplies', 'Tourism', 'Insurance', 'Sport and Leisure', 'Other', 'Additional Service', 
                                  'Fitness', 'Education', 'Direct Sales']
        prev_goods_cat = st.selectbox("PREV_NAME_GOODS_CATEGORY_<LAMBDA>", options=prev_goods_cat_options, index=prev_goods_cat_options.index("XNA"))
        
        prev_weekday_options = ['SATURDAY', 'FRIDAY', 'TUESDAY', 'MONDAY', 'SUNDAY', 'WEDNESDAY', 'THURSDAY']
        prev_weekday_lambda = st.selectbox("PREV_WEEKDAY_APPR_PROCESS_START_<LAMBDA>", options=prev_weekday_options, index=prev_weekday_options.index("TUESDAY"))

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
            st.error(f"Risk Level: High Risk (Defaulter)")
            result_text = "High Risk (Defaulter)"
        else:
            st.success(f"Risk Level: Low Risk (Non-Defaulter)")
            result_text = "Low Risk (Non-Defaulter)"
            
        st.write(f"**Probability:** {probability:.2%}")
        st.write(f"**Raw Score:** {raw_prediction:.4f}")


        if st.session_state.get('chat_session'):
            system_log_message = (
                f"SYSTEM LOG: New user loan data received. "
                f"Inputs: Loan Credit={input_data['AMT_CREDIT']}, "
                f"Income={input_data['AMT_INCOME_TOTAL']}, "
                f"Ext. Score 3={input_data['EXT_SOURCE_3']}, "
                f"Ext. Score 2={input_data['EXT_SOURCE_2']}, "
                f"Years Employed={input_data['YEARS_EMPLOYED']} years, "
                f"Years ID Publish={input_data['YEARS_ID_PUBLISH']} years, "
                f"Years Registration={input_data['YEARS_REGISTRATION']} years, "
                f"Years Birth={input_data['YEARS_BIRTH']} years, "
                f"Annuity={input_data['AMT_ANNUITY']}, "
                f"Goods Price={input_data['AMT_GOODS_PRICE']}, "
                f"Organization Type={input_data['ORGANIZATION_TYPE']}, "
                f"Occupation Type={input_data['OCCUPATION_TYPE']}, "
                f"Family Members={input_data['CNT_FAM_MEMBERS']}, "
                f"Region Population Relative={input_data['REGION_POPULATION_RELATIVE']}, "
                f"Own Car Age={input_data['OWN_CAR_AGE']}, "
                f"ID={input_data['SK_ID_CURR']}, "
                f"Results: Risk={result_text}, Probability={probability:.2%}, Raw Score={raw_prediction:.4f}."
                f"Please acknowledge this data but **do not respond** to the user at this time."
            )
            
            st.session_state.chat_session.send_message(system_log_message)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

st.markdown("---")
st.header("🤖 Prediction Analysis Chatbot")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the prediction results or general loan questions..."):
    
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.get('chat_session'):
        try:
            response = st.session_state.chat_session.send_message(prompt)
            bot_response = response.text
            
        except Exception as e:
            error_detail = str(e)
            if "rate limit" in error_detail.lower() or "quota" in error_detail.lower():
                 bot_response = "I've reached the usage limit. Please try later. 📉"
            else:
                 bot_response = f"An unexpected error occurred: {error_detail}"

        with st.chat_message("assistant"):
            st.markdown(bot_response)
            st.session_state.chat_messages.append({"role": "assistant", "content": bot_response})
    else:
        with st.chat_message("assistant"):
            st.markdown("Chatbot is currently offline due to initialization error. Check console/API Key.")
