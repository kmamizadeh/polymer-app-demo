import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Paths to the files. These should be relative to the script location.
EXCEL_FILE = 'Polymer_Properties_Processed_by_python1.xlsx'
IMPACT_MODEL_FILE = 'regression_model.pkl'
TENSILE_MODEL_FILE = 'tensile_model.pkl'

# Define standard units for the properties
UNITS_IMPACT = {
    "KJ/m": "kJ/m",
    "J/m": "J/m",
    "J/m^2": "J/m^2",
    "KJ/m^2": "KJ/m^2",
    "J/cm^2": "J/cm^2"
}
UNITS_TENSILE = {
    "Mpa": "MPa",
    "Gpa": "GPa",
    "Pa": "Pa"
}


# --- Data and Model Loading (with caching) ---
@st.cache_data
def load_data_and_get_unique_values():
    if not os.path.exists(EXCEL_FILE):
        return None, None

    df = pd.read_excel(EXCEL_FILE)

    unique_values = {
        'Polymer1_Type': sorted(df['Polymer1_Type'].unique()),
        'Polymer2_Type': sorted(df['Polymer2_Type'].unique()),
        'Polymer3_Type': sorted(df['Polymer3_Type'].unique()),
        'Filler1_Type': sorted(df['Filler1_Type'].unique()),
        'Filler2_Type': sorted(df['Filler2_Type'].unique()),
        'Additive_Type': sorted(df['Additive_Type'].unique())
    }

    return df, unique_values


@st.cache_resource
def load_model_and_get_columns():
    try:
        if not os.path.exists(IMPACT_MODEL_FILE) or not os.path.exists(TENSILE_MODEL_FILE):
            st.error("فایل های مدل (.pkl) پیدا نشدند. لطفا مطمئن شوید که در مسیر صحیح قرار دارند.")
            return None, None, None, None

        impact_model = joblib.load(IMPACT_MODEL_FILE)
        tensile_model = joblib.load(TENSILE_MODEL_FILE)

        impact_model_columns = impact_model.feature_names_in_.tolist()
        tensile_model_columns = tensile_model.feature_names_in_.tolist()

        return impact_model, tensile_model, impact_model_columns, tensile_model_columns

    except FileNotFoundError:
        st.error("خطا: فایل‌های مدل پیدا نشدند.")
        return None, None, None, None
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل: {e}")
        return None, None, None, None


# --- Main Functions ---
def convert_impact_to_base(value, unit):
    if unit == "J/m":
        return value
    elif unit == "KJ/m":
        return value * 1000
    elif unit in ["J/m^2", "KJ/m^2", "J/cm^2"]:
        # The model is trained on J/m, so we cannot convert the units directly without geometry data.
        # We will use the input value as is.
        st.warning("توجه: این تبدیل تقریبی است، زیرا اطلاعات ابعاد نمونه در دسترس نیست.")
        return value
    return value


def convert_tensile_to_base(value, unit):
    if unit == "MPa":
        return value
    elif unit == "GPa":
        return value * 1000
    elif unit == "Pa":
        return value / 1000000
    return value


# --- Prediction Logic ---
def predict_properties(data_to_predict, impact_model, tensile_model, impact_cols, tensile_cols):
    try:
        # Create a dataframe for impact prediction with all required columns
        df_impact = pd.DataFrame(columns=impact_cols)
        df_impact.loc[0] = 0

        # Create a dataframe for tensile prediction with all required columns
        df_tensile = pd.DataFrame(columns=tensile_cols)
        df_tensile.loc[0] = 0

        # Populate the dataframes from the input dictionary
        for key, value in data_to_predict.items():
            if value is None:
                continue

            if key in df_impact.columns:
                df_impact.loc[0, key] = value

            if key in df_tensile.columns:
                df_tensile.loc[0, key] = value

            if isinstance(value, str):
                # One-hot encode the categorical variables
                if f'{key}_{value}' in df_impact.columns:
                    df_impact.loc[0, f'{key}_{value}'] = 1
                if f'{key}_{value}' in df_tensile.columns:
                    df_tensile.loc[0, f'{key}_{value}'] = 1

        # Drop the original categorical columns from the dataframes before prediction
        categorical_cols = ['Polymer1_Type', 'Polymer2_Type', 'Polymer3_Type',
                            'Filler1_Type', 'Filler2_Type', 'Additive_Type',
                            'Impact_Test_Type']

        for col in categorical_cols:
            if col in df_impact.columns:
                df_impact = df_impact.drop(columns=[col])
            if col in df_tensile.columns:
                df_tensile = df_tensile.drop(columns=[col])

        # Make predictions
        impact_pred = impact_model.predict(df_impact)[0]
        tensile_pred = tensile_model.predict(df_tensile)[0]

        return {'impact': impact_pred, 'tensile': tensile_pred}
    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")
        return None


# --- Main App Structure ---
st.set_page_config(layout="wide", page_title="Polymer Property Predictor")
st.title("برنامه پیش‌بینی و ثبت خواص کامپوزیت‌های پلیمری")
st.markdown(
    """
    این برنامه به شما امکان می‌دهد خواص فرمولاسیون‌های پلیمری را ثبت کنید
    و بر اساس مدل‌های هوش مصنوعی، خواص نهایی آن‌ها را پیش‌بینی نمایید.
    """, unsafe_allow_html=True)

df, unique_values = load_data_and_get_unique_values()
impact_model, tensile_model, impact_cols, tensile_cols = load_model_and_get_columns()

col_form, col_predict = st.columns([1.5, 1])

with col_form:
    st.header("ثبت اطلاعات در دیتاست")

    with st.form(key='data_entry_form'):
        st.subheader("۱. مشخصات فرمولاسیون")

        st.markdown("---")

        st.markdown("**پلیمرها**")
        p1_type = st.selectbox("نوع پلیمر اول", options=[''] + unique_values['Polymer1_Type'], key="p1_type")
        p1_perc = st.number_input("درصد پلیمر اول", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p1_perc")
        p2_type = st.selectbox("نوع پلیمر دوم", options=[''] + unique_values['Polymer2_Type'], key="p2_type")
        p2_perc = st.number_input("درصد پلیمر دوم", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p2_perc")
        p3_type = st.selectbox("نوع پلیمر سوم", options=[''] + unique_values['Polymer3_Type'], key="p3_type")
        p3_perc = st.number_input("درصد پلیمر سوم", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="p3_perc")

        st.markdown("---")

        st.markdown("**فیلرها**")
        f1_type = st.selectbox("نوع فیلر اول", options=[''] + unique_values['Filler1_Type'], key="f1_type")
        f1_size = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, key="f1_size")
        f1_perc = st.number_input("درصد فیلر اول", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f1_perc")
        f2_type = st.selectbox("نوع فیلر دوم", options=[''] + unique_values['Filler2_Type'], key="f2_type")
        f2_size = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, key="f2_size")
        f2_perc = st.number_input("درصد فیلر دوم", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="f2_perc")

        st.markdown("---")

        st.markdown("**افزودنی‌ها**")
        a_type = st.selectbox("نوع افزودنی", options=[''] + unique_values['Additive_Type'], key="a_type")
        a_perc = st.number_input("درصد افزودنی", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="a_perc")
        a_func = st.selectbox("عملکرد افزودنی",
                              options=[''] + ['Toughener', 'Impact Modifier', 'Colorant', 'Antioxidant', 'Unknown'],
                              key="a_func")

        st.markdown("---")

        st.subheader("۲. نوع آزمون")
        impact_test_type = st.selectbox("نوع آزمون ضربه", options=[''] + ['Charpy', 'Izod'],
                                        key="impact_test_type_entry")
        impact_not_break = st.checkbox("شکسته نشد (No break)", key="impact_not_break")

        st.markdown("---")

        st.subheader("۳. خواص نهایی")
        impact_value = st.number_input(f"خواص ضربه (J/m)", min_value=0.0, key="impact_value")
        tensile_value = st.number_input("استحکام کششی (MPa)", min_value=0.0, key="tensile_value")

        submit_button = st.form_submit_button(label='ثبت اطلاعات')

        if submit_button:
            if df is not None:
                new_data = {
                    "Polymer1_Type": p1_type, "Polymer1_Perc": p1_perc,
                    "Polymer2_Type": p2_type, "Polymer2_Perc": p2_perc,
                    "Polymer3_Type": p3_type, "Polymer3_Perc": p3_perc,
                    "Filler1_Type": f1_type, "Filler1_ParticleSize_um": f1_size, "Filler1_Perc": f1_perc,
                    "Filler2_Type": f2_type, "Filler2_ParticleSize_um": f2_size, "Filler2_Perc": f2_perc,
                    "Additive_Type": a_type, "Additive_Perc": a_perc, "Additive_Functionality": a_func,
                    "Impact_Test_Type": impact_test_type, "Impact_Not_Break": impact_not_break,
                    "Impact_Value_Jm": convert_impact_to_base(impact_value, "J/m"),
                    "Tensile_Value_MPa": convert_tensile_to_base(tensile_value, "MPa")
                }

                new_row = pd.DataFrame([new_data])
                updated_df = pd.concat([df, new_row], ignore_index=True)
                updated_df.to_excel(EXCEL_FILE, index=False)
                st.success("اطلاعات با موفقیت ثبت شد!")
            else:
                st.error("خطا: فایل اکسل دیتاست پیدا نشد. لطفاً آن را آپلود کنید.")

with col_predict:
    st.header("پیش‌بینی خواص")

    with st.form(key='prediction_form'):
        st.subheader("مشخصات فرمولاسیون")

        st.markdown("---")

        st.markdown("**پلیمرها**")
        p1_type_p = st.selectbox("نوع پلیمر اول", options=[''] + unique_values['Polymer1_Type'], key="p1_type_p")
        p1_perc_p = st.number_input("درصد پلیمر اول", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="p1_perc_p")
        p2_type_p = st.selectbox("نوع پلیمر دوم", options=[''] + unique_values['Polymer2_Type'], key="p2_type_p")
        p2_perc_p = st.number_input("درصد پلیمر دوم", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="p2_perc_p")
        p3_type_p = st.selectbox("نوع پلیمر سوم", options=[''] + unique_values['Polymer3_Type'], key="p3_type_p")
        p3_perc_p = st.number_input("درصد پلیمر سوم", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="p3_perc_p")

        st.markdown("---")

        st.markdown("**فیلرها**")
        f1_type_p = st.selectbox("نوع فیلر اول", options=[''] + unique_values['Filler1_Type'], key="f1_type_p")
        f1_size_p = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, key="f1_size_p")
        f1_perc_p = st.number_input("درصد فیلر اول", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="f1_perc_p")
        f2_type_p = st.selectbox("نوع فیلر دوم", options=[''] + unique_values['Filler2_Type'], key="f2_type_p")
        f2_size_p = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, key="f2_size_p")
        f2_perc_p = st.number_input("درصد فیلر دوم", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                    key="f2_perc_p")

        st.markdown("---")

        st.markdown("**افزودنی‌ها**")
        a_type_p = st.selectbox("نوع افزودنی", options=[''] + unique_values['Additive_Type'], key="a_type_p")
        a_perc_p = st.number_input("درصد افزودنی", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="a_perc_p")
        a_func_p = st.selectbox("عملکرد افزودنی",
                                options=[''] + ['Toughener', 'Impact Modifier', 'Colorant', 'Antioxidant', 'Unknown'],
                                key="a_func_p")

        st.markdown("---")

        st.subheader("نوع آزمون")
        impact_test_type_p = st.selectbox("نوع آزمون ضربه", options=[''] + ['Charpy', 'Izod', 'Unknown'],
                                          key="impact_test_type_p")
        impact_not_break_p = st.checkbox("شکسته نشد (No break)", key="impact_not_break_p")

        predict_button = st.form_submit_button(label='پیش‌بینی خواص')

        if predict_button:
            if impact_model is not None and tensile_model is not None:
                data_to_predict = {
                    'Polymer1_Type': p1_type_p, 'Polymer1_Perc': p1_perc_p,
                    'Polymer2_Type': p2_type_p, 'Polymer2_Perc': p2_perc_p,
                    'Polymer3_Type': p3_type_p, 'Polymer3_Perc': p3_perc_p,
                    'Filler1_Type': f1_type_p, 'Filler1_ParticleSize_um': f1_size_p, 'Filler1_Perc': f1_perc_p,
                    'Filler2_Type': f2_type_p, 'Filler2_ParticleSize_um': f2_size_p, 'Filler2_Perc': f2_perc_p,
                    'Additive_Type': a_type_p, 'Additive_Perc': a_perc_p, 'Additive_Functionality': a_func_p,
                    'Impact_Test_Type': impact_test_type_p, 'Impact_Not_Break': impact_not_break_p
                }

                predictions = predict_properties(data_to_predict, impact_model, tensile_model, impact_cols,
                                                 tensile_cols)

                if predictions:
                    st.success("پیش‌بینی با موفقیت انجام شد!")

                    st.subheader("نتایج پیش‌بینی")
                    st.write(f"**خواص ضربه:** {predictions['impact']:.2f} J/m²")
                    st.write(f"**استحکام کششی:** {predictions['tensile']:.2f} MPa")
                else:
                    st.error("پیش‌بینی انجام نشد. لطفاً ورودی‌های خود را بررسی کنید.")
            else:
                st.warning("فایل‌های مدل پیدا نشدند. لطفاً آن‌ها را در کنار فایل app.py قرار دهید.")
