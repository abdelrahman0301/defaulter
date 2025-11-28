import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import traceback

warnings.filterwarnings('ignore')

class LoanDefaultPredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.load_model()
        
        # Define the set of features exposed in the main form (17 key features)
        self.main_features = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'YEARS_BIRTH', 'YEARS_EMPLOYED', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'REGION_POPULATION_RELATIVE', 'DAYS_ID_PUBLISH',
            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY'
        ]

    def load_model(self):
        try:
            # Load the LightGBM model from the pkl file
            self.model = joblib.load('LoanDefaulter_LightGBM.pkl')
            
            # Extract feature names from the model (108 features)
            if hasattr(self.model, 'feature_name_'):
                self.features = self.model.feature_name_
            else:
                # Fallback list (truncated for brevity, but includes all 108 if needed)
                self.features = [
                    'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'OWN_CAR_AGE', 'FLAG_MOBIL',
                    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
                    'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                    'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                    'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                    'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1',
                    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENT_AREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                    'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 
                    'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 
                    'BASEMENT_AREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 
                    'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 
                    'LIVINGAREA_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENT_AREA_MEDI', 
                    'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 
                    'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAREA_MEDI', 
                    'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 
                    'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'AMT_REQ_CREDIT_BUREAU_MON', 
                    'AMT_REQ_CREDIT_BUREAU_YEAR', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                    'NFLAG_LAST_APPL_IN_TIME', 'NFLAG_INSURANCE_REJECT', 'NAME_CASH_LOAN_PURPOSE', 
                    'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_TYPE_SUITE_AP', 'NAME_CLIENT_TYPE', 
                    'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 
                    'SELLERPLACE_AREA', 'NAME_CONTRACT_STATUS', 'CNT_PAYMENT', 'SK_DPD', 'SK_DPD_DEF',
                    'YEARS_BEGINEXPLUATATION_AVG_diff', 'YEARS_BEGINEXPLUATATION_MODE_diff', 
                    'YEARS_BEGINEXPLUATATION_MEDI_diff', 'WALLSMATERIAL_MODE', 'FONDKAPREMONT_MODE' 
                ]
            
            # Identify the features that are hidden from the main form
            self.hidden_features = [f for f in self.features if f not in self.main_features]

        except FileNotFoundError:
            st.error("Model file 'LoanDefaulter_LightGBM.pkl' not found.")
            self.model = None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None

    def get_default_inputs(self):
        """Return default values for all 108 features (based on training set averages/modes)."""
        defaults = {}

        # ---------------------------------------------------------
        # Key Features (Updated by User Input)
        # ---------------------------------------------------------
        defaults['AMT_INCOME_TOTAL'] = 150000.0
        defaults['AMT_CREDIT'] = 500000.0
        defaults['AMT_ANNUITY'] = 25000.0
        defaults['AMT_GOODS_PRICE'] = 450000.0
        defaults['YEARS_BIRTH'] = -14000.0 # ~38 years old
        defaults['YEARS_EMPLOYED'] = -2000.0 # ~5.5 years employed
        defaults['EXT_SOURCE_2'] = 0.5
        defaults['EXT_SOURCE_3'] = 0.5
        defaults['REGION_POPULATION_RELATIVE'] = 0.015
        defaults['DAYS_ID_PUBLISH'] = -2500.0 # Days since ID was published

        # Categorical defaults
        defaults['NAME_CONTRACT_TYPE'] = 'Cash loans'
        defaults['CODE_GENDER'] = 'F'
        defaults['NAME_EDUCATION_TYPE'] = 'Secondary / secondary special'
        defaults['NAME_FAMILY_STATUS'] = 'Married'
        defaults['NAME_HOUSING_TYPE'] = 'House / apartment'
        defaults['FLAG_OWN_CAR'] = 'N'
        defaults['FLAG_OWN_REALTY'] = 'Y'

        # ---------------------------------------------------------
        # Advanced/Hidden Features (Defaulted here)
        # ---------------------------------------------------------
        defaults['SK_ID_CURR'] = 300000.0
        defaults['OWN_CAR_AGE'] = 10.0
        defaults['CNT_CHILDREN'] = 0.0
        defaults['CNT_FAM_MEMBERS'] = 2.0
        defaults['HOUR_APPR_PROCESS_START'] = 10.0
        defaults['REGION_RATING_CLIENT'] = 2.0
        defaults['REGION_RATING_CLIENT_W_CITY'] = 2.0

        # Boolean Flags (Most are 0 by default)
        defaults['FLAG_MOBIL'] = 1.0
        defaults['FLAG_EMP_PHONE'] = 1.0
        defaults['FLAG_WORK_PHONE'] = 0.0
        defaults['FLAG_CONT_MOBILE'] = 1.0
        defaults['FLAG_PHONE'] = 0.0
        defaults['FLAG_EMAIL'] = 0.0
        defaults['REG_REGION_NOT_LIVE_REGION'] = 0.0
        defaults['REG_REGION_NOT_WORK_REGION'] = 0.0
        defaults['LIVE_REGION_NOT_WORK_REGION'] = 0.0
        defaults['REG_CITY_NOT_LIVE_CITY'] = 0.0
        defaults['REG_CITY_NOT_WORK_CITY'] = 0.0
        defaults['LIVE_CITY_NOT_WORK_CITY'] = 0.0
        defaults['FLAG_DOCUMENT_3'] = 1.0
        defaults['FLAG_DOCUMENT_6'] = 0.0
        defaults['FLAG_DOCUMENT_8'] = 0.0

        # Time-related defaults (in days)
        defaults['DAYS_REGISTRATION'] = -3500.0
        defaults['DAYS_LAST_PHONE_CHANGE'] = -1000.0

        # Social Circle defaults
        defaults['OBS_30_CNT_SOCIAL_CIRCLE'] = 2.0
        defaults['DEF_30_CNT_SOCIAL_CIRCLE'] = 0.0
        defaults['OBS_60_CNT_SOCIAL_CIRCLE'] = 2.0
        defaults['DEF_60_CNT_SOCIAL_CIRCLE'] = 0.0

        # Bureau (Credit history) defaults
        defaults['AMT_REQ_CREDIT_BUREAU_MON'] = 0.0
        defaults['AMT_REQ_CREDIT_BUREAU_YEAR'] = 1.0

        # Application Process/Previous Application defaults (These are synthetic/complex)
        defaults['NFLAG_LAST_APPL_IN_TIME'] = 1.0
        defaults['NFLAG_INSURANCE_REJECT'] = 0.0
        defaults['SELLERPLACE_AREA'] = 500.0
        defaults['CNT_PAYMENT'] = 24.0
        defaults['SK_DPD'] = 0.0
        defaults['SK_DPD_DEF'] = 0.0
        defaults['YEARS_BEGINEXPLUATATION_AVG_diff'] = 0.0
        defaults['YEARS_BEGINEXPLUATATION_MODE_diff'] = 0.0
        defaults['YEARS_BEGINEXPLUATATION_MEDI_diff'] = 0.0

        # Categorical defaults for hidden features
        defaults['NAME_TYPE_SUITE'] = 'Unaccompanied'
        defaults['NAME_INCOME_TYPE'] = 'Working'
        defaults['OCCUPATION_TYPE'] = 'Laborers'
        defaults['ORGANIZATION_TYPE'] = 'Business Entity Type 3'
        defaults['WEEKDAY_APPR_PROCESS_START'] = 'TUESDAY'
        defaults['WALLSMATERIAL_MODE'] = 'Panel'
        defaults['FONDKAPREMONT_MODE'] = 'reg oper account'
        
        # Previous Application defaults (synthetic)
        defaults['NAME_CASH_LOAN_PURPOSE'] = 'XNA'
        defaults['NAME_PAYMENT_TYPE'] = 'Cash through the bank'
        defaults['CODE_REJECT_REASON'] = 'XNA'
        defaults['NAME_TYPE_SUITE_AP'] = 'Unaccompanied'
        defaults['NAME_CLIENT_TYPE'] = 'Repeater'
        defaults['NAME_GOODS_CATEGORY'] = 'XNA'
        defaults['NAME_PORTFOLIO'] = 'Cash'
        defaults['NAME_PRODUCT_TYPE'] = 'XNA'
        defaults['CHANNEL_TYPE'] = 'Credit and cash offices'
        defaults['NAME_CONTRACT_STATUS'] = 'Approved'

        # Apartment/Housing features (Set to mean/median)
        defaults['APARTMENTS_AVG'] = 0.11
        defaults['BASEMENT_AREA_AVG'] = 0.08
        defaults['YEARS_BEGINEXPLUATATION_AVG'] = 0.97
        defaults['YEARS_BUILD_AVG'] = 0.75
        defaults['COMMONAREA_AVG'] = 0.02
        defaults['ELEVATORS_AVG'] = 0.07
        defaults['ENTRANCES_AVG'] = 0.15
        defaults['FLOORSMAX_AVG'] = 0.22
        defaults['FLOORSMIN_AVG'] = 0.19
        defaults['LANDAREA_AVG'] = 0.06
        defaults['LIVINGAREA_AVG'] = 0.11
        defaults['NONLIVINGAREA_AVG'] = 0.03
        
        # Mode/Medi values (simplified to match AVG for this demo)
        defaults['APARTMENTS_MODE'] = defaults['APARTMENTS_AVG']
        defaults['BASEMENT_AREA_MODE'] = defaults['BASEMENT_AREA_AVG']
        defaults['YEARS_BEGINEXPLUATATION_MODE'] = defaults['YEARS_BEGINEXPLUATATION_AVG']
        defaults['YEARS_BUILD_MODE'] = defaults['YEARS_BUILD_AVG']
        defaults['COMMONAREA_MODE'] = defaults['COMMONAREA_AVG']
        defaults['ELEVATORS_MODE'] = defaults['ELEVATORS_AVG']
        defaults['ENTRANCES_MODE'] = defaults['ENTRANCES_AVG']
        defaults['FLOORSMAX_MODE'] = defaults['FLOORSMAX_AVG']
        defaults['FLOORSMIN_MODE'] = defaults['FLOORSMIN_AVG']
        defaults['LANDAREA_MODE'] = defaults['LANDAREA_AVG']
        defaults['LIVINGAREA_MODE'] = defaults['LIVINGAREA_AVG']
        defaults['NONLIVINGAREA_MODE'] = defaults['NONLIVINGAREA_AVG']
        
        defaults['APARTMENTS_MEDI'] = defaults['APARTMENTS_AVG']
        defaults['BASEMENT_AREA_MEDI'] = defaults['BASEMENT_AREA_AVG']
        defaults['YEARS_BEGINEXPLUATATION_MEDI'] = defaults['YEARS_BEGINEXPLUATATION_AVG']
        defaults['YEARS_BUILD_MEDI'] = defaults['YEARS_BUILD_AVG']
        defaults['COMMONAREA_MEDI'] = defaults['COMMONAREA_AVG']
        defaults['ELEVATORS_MEDI'] = defaults['ELEVATORS_AVG']
        defaults['ENTRANCES_MEDI'] = defaults['ENTRANCES_AVG']
        defaults['FLOORSMAX_MEDI'] = defaults['FLOORSMAX_AVG']
        defaults['FLOORSMIN_MEDI'] = defaults['FLOORSMIN_AVG']
        defaults['LANDAREA_MEDI'] = defaults['LANDAREA_AVG']
        defaults['LIVINGAREA_MEDI'] = defaults['LIVINGAREA_AVG']
        defaults['NONLIVINGAREA_MEDI'] = defaults['NONLIVINGAREA_AVG']
        defaults['TOTALAREA_MODE'] = defaults['LIVINGAREA_MODE'] # Approximation
        
        # Document Flags (Most are 0)
        defaults['FLAG_DOCUMENT_2'] = 0.0
        defaults['FLAG_DOCUMENT_4'] = 0.0
        defaults['FLAG_DOCUMENT_5'] = 0.0
        defaults['FLAG_DOCUMENT_7'] = 0.0
        defaults['FLAG_DOCUMENT_9'] = 0.0
        defaults['FLAG_DOCUMENT_10'] = 0.0
        defaults['FLAG_DOCUMENT_11'] = 0.0
        defaults['FLAG_DOCUMENT_12'] = 0.0
        defaults['FLAG_DOCUMENT_13'] = 0.0
        defaults['FLAG_DOCUMENT_14'] = 0.0
        defaults['FLAG_DOCUMENT_15'] = 0.0
        defaults['FLAG_DOCUMENT_16'] = 0.0
        defaults['FLAG_DOCUMENT_17'] = 0.0
        defaults['FLAG_DOCUMENT_18'] = 0.0
        defaults['FLAG_DOCUMENT_19'] = 0.0
        defaults['FLAG_DOCUMENT_20'] = 0.0
        defaults['FLAG_DOCUMENT_21'] = 0.0

        return defaults

    def convert_to_model_format(self, user_inputs, advanced_inputs):
        """Combines default inputs with user inputs for the model."""
        
        # 1. Start with all 108 features set to their defaults
        model_inputs = self.get_default_inputs()
        
        # 2. Overwrite the 17 main features using the user's main form data
        model_inputs.update({
            # Financials
            'AMT_INCOME_TOTAL': user_inputs['income_total'],
            'AMT_CREDIT': user_inputs['credit_amount'],
            'AMT_ANNUITY': user_inputs['annuity'],
            'AMT_GOODS_PRICE': user_inputs['goods_price'],
            
            # Demographic/Credit Scores (Note: YEARS_BIRTH and YEARS_EMPLOYED are negative days)
            'YEARS_BIRTH': -user_inputs['age'] * 365.25,
            'YEARS_EMPLOYED': -user_inputs['years_employed'] * 365.25,
            'EXT_SOURCE_2': user_inputs['ext_source_2'],
            'EXT_SOURCE_3': user_inputs['ext_source_3'],
            'REGION_POPULATION_RELATIVE': user_inputs['region_pop'],
            'DAYS_ID_PUBLISH': -user_inputs['days_id_publish'],
            
            # Categorical
            'NAME_CONTRACT_TYPE': user_inputs['contract_type'],
            'CODE_GENDER': user_inputs['gender'],
            'NAME_EDUCATION_TYPE': user_inputs['education'],
            'NAME_FAMILY_STATUS': user_inputs['family_status'],
            'NAME_HOUSING_TYPE': user_inputs['housing_type'],
            'FLAG_OWN_CAR': user_inputs['own_car'],
            'FLAG_OWN_REALTY': user_inputs['own_realty'],
        })

        # 3. Overwrite any advanced features selected by the user
        model_inputs.update(advanced_inputs)
        
        return model_inputs

    def predict(self, input_data):
        if self.model is None:
            st.error("Model not loaded. Cannot make prediction.")
            return 0.5, False

        try:
            # 1. Organize data into a DataFrame with the features expected by the model
            # Ensure the data order matches the model's expected feature order
            ordered_data = [input_data.get(f, np.nan) for f in self.features]
            input_df = pd.DataFrame([ordered_data], columns=self.features)

            # 2. Robustly encode ALL string/object columns using the fixed hashing method
            string_columns = input_df.select_dtypes(include=['object']).columns

            # IMPORTANT: This must be a deterministic hash for consistency between runs/training
            def encode_as_int(x):
                # Use a simple, non-random hash function (adler32) for deterministic encoding
                import zlib
                return zlib.adler32(str(x).encode('utf-8')) & 0xFFFFFFFF
            
            for col in string_columns:
                input_df[col] = input_df[col].apply(encode_as_int)

            # 3. Final Conversion to numpy array for the model
            X = input_df.to_numpy(dtype=float)

            # 4. Predict
            default_prob = self.model.predict_proba(X)[0][1]
            return default_prob, default_prob > 0.5

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.code(traceback.format_exc())
            return 0.5, False


# --- Streamlit UI Code ---

# Utility function to create a meaningful label for the advanced features
def create_advanced_label(feature_name):
    # Mapping for some commonly confusing feature names
    mapping = {
        'YEARS_BIRTH': 'Age (Years) - **MUST** be positive value in years',
        'YEARS_EMPLOYED': 'Years Employed - **MUST** be positive value in years',
        'DAYS_REGISTRATION': 'Days Since Registration',
        'DAYS_LAST_PHONE_CHANGE': 'Days Since Last Phone Change',
        'CNT_CHILDREN': 'Number of Children',
        'CNT_FAM_MEMBERS': 'Number of Family Members',
        'FLAG_DOCUMENT_3': 'Has Document 3 (1=Yes, 0=No)',
        'FLAG_WORK_PHONE': 'Has Work Phone (1=Yes, 0=No)',
        'REGION_RATING_CLIENT': 'Region Rating (1=Best, 3=Worst)',
        'EXT_SOURCE_1': 'External Score 1 (0.0 to 1.0)',
        'OCCUPATION_TYPE': 'Occupation Type',
        'NAME_TYPE_SUITE': 'Client Accompanied By',
        'AMT_REQ_CREDIT_BUREAU_YEAR': 'Credit Bureau Inquiries (Last Year)',
    }
    return mapping.get(feature_name, feature_name.replace('_', ' ').title())


def app():
    st.set_page_config(page_title="Loan Default Predictor", layout="wide")
    st.title("💡 Loan Default Risk Predictor (LightGBM)")
    st.markdown("Adjust the applicant's parameters and press **Predict** to assess the credit risk.")

    predictor = LoanDefaultPredictor()
    if predictor.model is None:
        return

    # 1. Main Form Inputs (17 Features)
    st.subheader("1. Core Applicant Data")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Financial Inputs
        income_total = st.number_input("Annual Income (Total)", min_value=30000.0, max_value=5000000.0, value=150000.0, step=10000.0, key='income_total')
        credit_amount = st.number_input("Credit Amount (Loan)", min_value=50000.0, max_value=4000000.0, value=500000.0, step=10000.0, key='credit_amount')
        annuity = st.number_input("Credit Annuity", min_value=1000.0, max_value=200000.0, value=25000.0, step=1000.0, key='annuity')
        goods_price = st.number_input("Goods Price", min_value=50000.0, max_value=4000000.0, value=450000.0, step=10000.0, key='goods_price')

    with col2:
        # Demographic Inputs
        age = st.slider("Age (Years)", min_value=20, max_value=70, value=38, key='age')
        years_employed = st.slider("Years Employed", min_value=0, max_value=50, value=5, key='years_employed')
        
        # Binary flags
        own_car = st.selectbox("Owns Car?", ('N', 'Y'), key='own_car')
        own_realty = st.selectbox("Owns Realty?", ('Y', 'N'), key='own_realty')
        
        # Region info
        region_pop = st.number_input("Region Pop. Relative", min_value=0.0001, max_value=0.07, value=0.015, step=0.001, format="%.4f", key='region_pop')


    with col3:
        # Credit Score Inputs (External Sources)
        ext_source_2 = st.slider("External Score 2", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.3f", key='ext_source_2')
        ext_source_3 = st.slider("External Score 3", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.3f", key='ext_source_3')
        days_id_publish = st.slider("Days Since ID Published (Recency)", min_value=100, max_value=7000, value=2500, key='days_id_publish')
        
    with col4:
        # Categorical Inputs
        gender = st.selectbox("Gender", ('F', 'M', 'XNA'), key='gender')
        contract_type = st.selectbox("Contract Type", ('Cash loans', 'Revolving loans'), key='contract_type')
        education = st.selectbox("Education Type", ('Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'), key='education')
        family_status = st.selectbox("Family Status", ('Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'), key='family_status')
        housing_type = st.selectbox("Housing Type", ('House / apartment', 'Rented apartment', 'Municipal apartment', 'Office apartment', 'Co-op apartment', 'With parents'), key='housing_type')


    # 2. Advanced Features Dropdown (The required change)
    st.subheader("2. Advanced Feature Customization")
    
    # Use a session state to store selected advanced features so they don't reset on every interaction
    if 'selected_adv_features' not in st.session_state:
        st.session_state.selected_adv_features = []

    # Filter out features already present in the main form
    selectable_features = [f for f in predictor.hidden_features if f not in predictor.main_features]
    
    # Multi-select dropdown
    selected_features = st.multiselect(
        "Select additional features to customize (defaults will be used for unselected)",
        options=selectable_features,
        default=st.session_state.selected_adv_features
    )
    st.session_state.selected_adv_features = selected_features
    
    advanced_inputs = {}
    
    if selected_features:
        st.markdown("**Customize Selected Advanced Features:**")
        adv_cols = st.columns(min(len(selected_features), 4))
        
        # Loop through selected features and create input fields
        for i, feature in enumerate(selected_features):
            default_value = predictor.get_default_inputs().get(feature)
            column_type = type(default_value)
            label = create_advanced_label(feature)
            
            with adv_cols[i % 4]: # Place inputs in columns (max 4 per row)
                if column_type is float or column_type is int:
                    # Treat all numeric features as floats for flexibility
                    if feature == 'YEARS_BIRTH' or feature == 'YEARS_EMPLOYED':
                        # Special handling to show years instead of negative days
                        current_default = int(abs(default_value / 365.25)) if default_value else 0
                        input_value = st.number_input(
                            label=label, 
                            value=current_default, 
                            min_value=0, 
                            max_value=100,
                            key=f'adv_{feature}'
                        )
                        # Store in negative days format for the model
                        advanced_inputs[feature] = -input_value * 365.25
                    else:
                        input_value = st.number_input(
                            label=label, 
                            value=default_value if default_value is not None else 0.0, 
                            key=f'adv_{feature}'
                        )
                        advanced_inputs[feature] = input_value
                
                elif column_type is str:
                    # Simplified selection for categorical features (you might need specific options if a full list is known)
                    input_value = st.text_input(
                        label=label, 
                        value=default_value if default_value is not None else 'XNA', 
                        key=f'adv_{feature}',
                        help="Enter the category value. Common examples: 'Working', 'Laborers', 'Business Entity Type 3'."
                    )
                    advanced_inputs[feature] = input_value
                else:
                    st.warning(f"Feature type for '{feature}' not handled.")

    st.markdown("---")
    
    if st.button("Predict Loan Default Risk", type="primary"):
        
        # Convert user-friendly inputs to model-compatible format
        user_inputs = {
            'income_total': income_total, 'credit_amount': credit_amount, 'annuity': annuity, 
            'goods_price': goods_price, 'age': age, 'years_employed': years_employed, 
            'ext_source_2': ext_source_2, 'ext_source_3': ext_source_3, 'region_pop': region_pop,
            'days_id_publish': days_id_publish, 'contract_type': contract_type, 
            'gender': gender, 'education': education, 'family_status': family_status, 
            'housing_type': housing_type, 'own_car': own_car, 'own_realty': own_realty
        }

        # Combine main and advanced inputs
        input_data = predictor.convert_to_model_format(user_inputs, advanced_inputs)
        
        # Run Prediction
        default_prob, will_default = predictor.predict(input_data)
        
        st.header("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        # Extract the user-friendly age and years employed from input_data for display
        display_age = int(abs(input_data.get('YEARS_BIRTH', 0) / 365.25))
        display_employed = int(abs(input_data.get('YEARS_EMPLOYED', 0) / 365.25))

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
            st.success("🎉 **LOW RISK**: This applicant shows strong creditworthiness with low probability of default.")
        elif default_prob < 0.6:
            st.warning("⚠️ **MEDIUM RISK**: This applicant has moderate risk factors. Additional review or manual underwriting recommended.")
        else:
            st.error("🚨 **HIGH RISK**: This applicant shows significant risk factors for default. Proceed with caution or decline.")
        
        with st.expander("Technical Details: Features Sent to Model"):
            st.write("The model received a total of 108 features. Here are the **customized or key** values:")
            
            # Display all user-set inputs, including advanced ones
            key_values = {
                'Age (Years)': display_age,
                'Employment (Years)': display_employed,
                'Income': f"${input_data.get('AMT_INCOME_TOTAL'):,.0f}",
                'Credit Score 2': f"{input_data.get('EXT_SOURCE_2'):.3f}",
                'Credit Score 3': f"{input_data.get('EXT_SOURCE_3'):.3f}",
                'Contract Type': input_data.get('NAME_CONTRACT_TYPE'),
            }
            # Add all customized advanced inputs to the display list
            for k, v in advanced_inputs.items():
                key_values[create_advanced_label(k)] = v

            st.json(key_values)
            st.caption("Note: All unlisted features were set to their default values (e.g., all document flags set to 0).")


if __name__ == "__main__":
    app()
