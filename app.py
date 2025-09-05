import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np

# Paths to the files. These should be relative to the script location.
EXCEL_FILE = 'Polymer_Properties_Processed_by_python1.xlsx'
IMPACT_MODEL_FILE = 'regression_model.pkl'
TENSILE_MODEL_FILE = 'tensile_model.pkl'

# --- تعریف واحدهای استاندارد و ضرایب تبدیل ---
# واحد پایه برای Impact: J/m
CHARPY_UNITS = {
    "J/m^2": 1.0,
    "kJ/m^2": 1000.0,
}

IZOD_UNITS = {
    "J/m": 1.0,
    "ft-lb/in": 53.3787
}

# واحد پایه برای Tensile Strength: MPa
TENSILE_UNITS = {
    "MPa": 1.0,
    "psi": 0.00689476,
    "GPa": 1000.0
}


# --- توابع تبدیل واحد ---
def convert_impact_to_base(value, unit, test_type):
    """مقدار ضربه را به واحد پایه (J/m) تبدیل می‌کند."""
    if test_type == "Charpy":
        if unit in CHARPY_UNITS:
            return value * CHARPY_UNITS[unit]
    elif test_type == "Izod":
        if unit in IZOD_UNITS:
            return value * IZOD_UNITS[unit]
    return value


def convert_tensile_to_base(value, unit):
    """مقدار استحکام کششی را به واحد پایه (MPa) تبدیل می‌کند."""
    if unit in TENSILE_UNITS:
        return value * TENSILE_UNITS[unit]
    return value


# --- بارگذاری دیتاست و استخراج مقادیر یکتا ---
@st.cache_data
def load_data_and_get_unique_values():
    """دیتاست را بارگذاری و مقادیر یکتا را برای استفاده در منوها استخراج می‌کند."""
    if not os.path.exists(EXCEL_FILE):
        st.error(f"فایل اکسل دیتاست با مسیر {EXCEL_FILE} پیدا نشد.")
        return {}, pd.DataFrame(), False

    try:
        df = pd.read_excel(EXCEL_FILE)

        # استخراج مقادیر یکتا از ستون‌های مربوطه
        all_polymers = pd.concat(
            [df['Polymer1_Type'], df['Polymer2_Type'], df['Polymer3_Type']]).dropna().unique().tolist()
        all_additives = df['Additive_Type'].dropna().unique().tolist()
        all_fillers = pd.concat([df['Filler1_Type'], df['Filler2_Type']]).dropna().unique().tolist()

        # Adding 'None' option to the beginning of the lists
        unique_values = {
            'All_Polymers': ['None'] + sorted(all_polymers),
            'All_Additives': ['None'] + sorted(all_additives),
            'All_Fillers': ['None'] + sorted(all_fillers),
            'Impact_Test_Type': ['نامشخص', 'Charpy', 'Izod']
        }

        return unique_values, df, True
    except Exception as e:
        st.error(f"خطا در خواندن فایل اکسل: {e}")
        return {}, pd.DataFrame(), False


# --- بارگذاری مدل و لیست ستون‌های ویژگی ---
@st.cache_resource
def load_model_and_get_columns():
    """مدل آموزش‌دیده را بارگذاری می‌کند."""
    try:
        impact_model = joblib.load(IMPACT_MODEL_FILE)
        # Assuming the model has a feature_names_in_ attribute
        training_columns_impact = impact_model.feature_names_in_
    except FileNotFoundError:
        st.error(f"فایل مدل (regression_model.pkl) با مسیر {IMPACT_MODEL_FILE} پیدا نشد.")
        return None, None, None, None

    try:
        tensile_model = joblib.load(TENSILE_MODEL_FILE)
        training_columns_tensile = tensile_model.feature_names_in_
    except FileNotFoundError:
        st.warning(
            f"فایل مدل برای پیش‌بینی استحکام کششی ({TENSILE_MODEL_FILE}) پیدا نشد. تنها پیش‌بینی خواص ضربه فعال است.")
        tensile_model = None
        training_columns_tensile = None

    return impact_model, tensile_model, training_columns_impact, training_columns_tensile


# تابع برای پیش‌بینی
def predict_properties(impact_model, tensile_model, data_dict, training_columns_impact, training_columns_tensile):
    """
    داده‌های ورودی را پردازش، همگام‌سازی و سپس پیش‌بینی را انجام می‌دهد.
    این تابع یک دیتافریم با فرمت صحیح برای مدل ایجاد می‌کند.
    """
    predictions = {}

    if impact_model:
        df_impact_predict = pd.DataFrame(np.zeros((1, len(training_columns_impact))), columns=training_columns_impact)
        for key, value in data_dict.items():
            if value is not None:
                # Handle one-hot encoded features for impact model
                if key in ['Polymer1_Type', 'Polymer2_Type', 'Polymer3_Type', 'Filler1_Type', 'Filler2_Type',
                           'Additive_Type', 'Impact_Test_Type']:
                    if value != 'None' and value != 'نامشخص':
                        col_name = f"{key}_{value}"
                        if col_name in df_impact_predict.columns:
                            df_impact_predict[col_name] = 1
                # Handle numeric features for impact model
                elif key in df_impact_predict.columns:
                    df_impact_predict[key] = value

        impact_prediction = impact_model.predict(df_impact_predict)
        predictions['impact'] = impact_prediction[0]

    if tensile_model:
        df_tensile_predict = pd.DataFrame(np.zeros((1, len(training_columns_tensile))),
                                          columns=training_columns_tensile)
        for key, value in data_dict.items():
            if value is not None:
                # Handle one-hot encoded features for tensile model
                if key in ['Polymer1_Type', 'Polymer2_Type', 'Polymer3_Type', 'Filler1_Type', 'Filler2_Type',
                           'Additive_Type']:
                    if value != 'None' and value != 'نامشخص':
                        col_name = f"{key}_{value}"
                        if col_name in df_tensile_predict.columns:
                            df_tensile_predict[col_name] = 1
                # Handle numeric features for tensile model
                elif key in df_tensile_predict.columns:
                    df_tensile_predict[key] = value

        tensile_prediction = tensile_model.predict(df_tensile_predict)
        predictions['tensile'] = tensile_prediction[0]

    return predictions


# --- ساخت واسط کاربری (UI) با Streamlit ---
unique_values, df_data, data_loaded = load_data_and_get_unique_values()
impact_model, tensile_model, training_columns_impact, training_columns_tensile = load_model_and_get_columns()

if data_loaded and impact_model and training_columns_impact is not None:
    st.set_page_config(layout="wide")
    st.title("ثبت و پیش‌بینی خواص فرمولاسیون‌های پلیمری")

    st.image(
        "https://www.freepik.com/premium-photo/polymer-pellets-plastic-resin-background-polypropylene-polyethylene-pvc-plastic-granules_15830230.htm",
        use_container_width=True)

    col_form, col_predict = st.columns([1, 1])

    # --- ستون سمت چپ: فرم ثبت اطلاعات ---
    with col_form:
        st.header("ثبت اطلاعات جدید")

        # --- بخش اول: مشخصات فرمولاسیون ---
        st.subheader("مشخصات فرمولاسیون")

        with st.container():
            p1_type = st.text_input("نوع پلیمر اول", key="p1t")
            p1_perc = st.number_input("درصد پلیمر اول (%)", min_value=0.0, max_value=100.0, step=0.1, key="p1p")

            p2_type = st.text_input("نوع پلیمر دوم", key="p2t")
            p2_perc = st.number_input("درصد پلیمر دوم (%)", min_value=0.0, max_value=100.0, step=0.1, key="p2p")

            p3_type = st.text_input("نوع پلیمر سوم", key="p3t")
            p3_perc = st.number_input("درصد پلیمر سوم (%)", min_value=0.0, max_value=100.0, step=0.1, key="p3p")

            st.subheader("فیلرها و افزودنی‌ها")
            f1_type = st.text_input("نوع فیلر اول", key="f1t")
            f1_perc = st.number_input("درصد فیلر اول (%)", min_value=0.0, max_value=100.0, step=0.1, key="f1p")
            f1_size = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, step=0.1, key="f1s")

            f2_type = st.text_input("نوع فیلر دوم", key="f2t")
            f2_perc = st.number_input("درصد فیلر دوم (%)", min_value=0.0, max_value=100.0, step=0.1, key="f2p")
            f2_size = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, step=0.1, key="f2s")

            a_type = st.text_input("نوع افزودنی", key="at")
            a_perc = st.number_input("درصد افزودنی (%)", min_value=0.0, max_value=100.0, step=0.1, key="ap")
            a_func = st.text_input("عملکرد افزودنی", key="af")

        # --- بخش دوم: نوع آزمون ضربه (خارج از فرم برای به‌روزرسانی پویا) ---
        st.markdown("---")
        st.subheader("نوع آزمون ضربه")
        impact_test_type_reg = st.selectbox("نوع آزمون ضربه", options=unique_values['Impact_Test_Type'], key="itt_reg")

        # منطق پویا برای به‌روزرسانی واحدهای ضربه
        units_options = []
        if impact_test_type_reg == 'Charpy':
            units_options = list(CHARPY_UNITS.keys())
        elif impact_test_type_reg == 'Izod':
            units_options = list(IZOD_UNITS.keys())

        # --- بخش سوم: خواص نهایی و دکمه ثبت ---
        st.markdown("---")
        st.subheader("خواص نهایی")

        with st.form("final_properties_form"):
            col_impact_val, col_impact_unit = st.columns([0.7, 0.3])
            with col_impact_val:
                impact_strength = st.number_input("خواص ضربه", min_value=0.0, key="is")
            with col_impact_unit:
                impact_unit = st.selectbox("واحد", options=units_options, key="iu")

            impact_not_break = st.checkbox("آزمون منجر به شکست نشد؟ (Not break)", key="inb")

            col_tensile_val, col_tensile_unit = st.columns([0.7, 0.3])
            with col_tensile_val:
                tensile_strength = st.number_input("خواص استحکام کششی", min_value=0.0, key="ts")
            with col_tensile_unit:
                tensile_unit = st.selectbox("واحد", list(TENSILE_UNITS.keys()), key="tu")

            st.markdown("---")
            submitted = st.form_submit_button("ثبت اطلاعات در دیتاست")

        if submitted:
            total_percent = p1_perc + p2_perc + p3_perc + f1_perc + f2_perc + a_perc
            if abs(total_percent - 100) > 0.01:
                st.error(f"مجموع درصدها باید 100 باشد، اما {total_percent} است. لطفا مقادیر را اصلاح کنید.")
            else:
                converted_impact_strength = None
                if impact_test_type_reg != 'نامشخص':
                    if impact_not_break:
                        converted_impact_strength = np.nan
                    elif impact_strength is not None and impact_unit is not None:
                        converted_impact_strength = convert_impact_to_base(impact_strength, impact_unit,
                                                                           impact_test_type_reg)

                converted_tensile_strength = convert_tensile_to_base(tensile_strength, tensile_unit)

                data = {
                    'Polymer1_Type': p1_type, 'Polymer1_Perc': p1_perc,
                    'Polymer2_Type': p2_type, 'Polymer2_Perc': p2_perc,
                    'Polymer3_Type': p3_type, 'Polymer3_Perc': p3_perc,
                    'Filler1_Type': f1_type, 'Filler1_ParticleSize_um': f1_size, 'Filler1_Perc': f1_perc,
                    'Filler2_Type': f2_type, 'Filler2_ParticleSize_um': f2_size, 'Filler2_Perc': f2_perc,
                    'Additive_Type': a_type, 'Additive_Perc': a_perc, 'Additive_Functionality': a_func,
                    'Impact_Value': converted_impact_strength, 'Tensile_Strength': converted_tensile_strength,
                    'Impact_Unit': impact_unit, 'Tensile_Unit': tensile_unit,
                    'Impact_Not_Break': impact_not_break,
                    'Impact_Test_Type': impact_test_type_reg
                }

                if os.path.exists(EXCEL_FILE):
                    existing_df = pd.read_excel(EXCEL_FILE)
                    updated_df = pd.concat([existing_df, pd.DataFrame([data])], ignore_index=True)
                else:
                    updated_df = pd.DataFrame([data])
                updated_df.to_excel(EXCEL_FILE, index=False)
                st.success("اطلاعات با موفقیت ثبت شد و در فایل اکسل ذخیره گردید.")

    # --- نمایش دیتاست کامل ---
    st.markdown("---")
    st.header("نمایش دیتاست")
    st.dataframe(df_data)

    # --- ستون سمت راست: فرم پیش‌بینی ---
    with col_predict:
        st.header("پیش‌بینی خواص")
        with st.form("prediction_form"):
            st.subheader("وارد کردن مشخصات فرمولاسیون برای پیش‌بینی")

            p1_type_p = st.selectbox("نوع پلیمر اول (برای پیش‌بینی)", options=unique_values['All_Polymers'],
                                     key="p1t_p")
            p1_perc_p = st.number_input("درصد پلیمر اول (برای پیش‌بینی)", min_value=0.0, max_value=100.0, step=0.1,
                                        key="p1p_p")

            p2_type_p = st.selectbox("نوع پلیمر دوم (برای پیش‌بینی)", options=unique_values['All_Polymers'],
                                     key="p2t_p")
            p2_perc_p = st.number_input("درصد پلیمر دوم (برای پیش‌بینی)", min_value=0.0, max_value=100.0, step=0.1,
                                        key="p2p_p")

            p3_type_p = st.selectbox("نوع پلیمر سوم (برای پیش‌بینی)", options=unique_values['All_Polymers'],
                                     key="p3t_p")
            p3_perc_p = st.number_input("درصد پلیمر سوم (برای پیش‌بینی)", min_value=0.0, max_value=100.0, step=0.1,
                                        key="p3p_p")

            f1_type_p = st.selectbox("نوع فیلر اول (برای پیش‌بینی)", options=unique_values['All_Fillers'], key="f1t_p")
            f1_perc_p = st.number_input("درصد فیلر اول (برای پیش‌بینی)", min_value=0.0, max_value=100.0, step=0.1,
                                        key="f1p_p")
            f1_size_p = st.number_input("اندازه ذرات فیلر اول (میکرون)", min_value=0.0, step=0.1, key="f1s_p")

            f2_type_p = st.selectbox("نوع فیلر دوم (برای پیش‌بینی)", options=unique_values['All_Fillers'], key="f2t_p")
            f2_perc_p = st.number_input("درصد فیلر دوم (برای پیش‌بینی)", min_value=0.0, max_value=100.0, step=0.1,
                                        key="f2p_p")
            f2_size_p = st.number_input("اندازه ذرات فیلر دوم (میکرون)", min_value=0.0, step=0.1, key="f2s_p")

            a_type_p = st.selectbox("نوع افزودنی (برای پیش‌بینی)", options=unique_values['All_Additives'], key="at_p")
            a_perc_p = st.number_input("درصد افزودنی (%)", min_value=0.0, max_value=100.0, step=0.1, key="ap_p")
            a_func_p = st.text_input("عملکرد افزودنی (برای پیش‌بینی)", key="af_p")

            st.subheader("شرایط آزمون")
            impact_test_type_p = st.selectbox("نوع آزمون ضربه", options=unique_values['Impact_Test_Type'], key="itt_p")
            impact_not_break_p = st.checkbox("آزمون منجر به شکست نشد؟", key="inb_p")

            predict_button = st.form_submit_button("پیش‌بینی خواص")

        if predict_button:
            data_to_predict = {
                'Polymer1_Type': p1_type_p, 'Polymer1_Perc': p1_perc_p,
                'Polymer2_Type': p2_type_p, 'Polymer2_Perc': p2_perc_p,
                'Polymer3_Type': p3_type_p, 'Polymer3_Perc': p3_perc_p,
                'Filler1_Type': f1_type_p, 'Filler1_ParticleSize_um': f1_size_p, 'Filler1_Perc': f1_perc_p,
                'Filler2_Type': f2_type_p, 'Filler2_ParticleSize_um': f2_size_p, 'Filler2_Perc': f2_perc_p,
                'Additive_Type': a_type_p, 'Additive_Perc': a_perc_p, 'Additive_Functionality': a_func_p,
                'Impact_Test_Type': impact_test_type_p,
                'Impact_Not_Break': impact_not_break_p
            }

            try:
                predictions = predict_properties(impact_model, tensile_model, data_to_predict, training_columns_impact,
                                                 training_columns_tensile)
                st.subheader("نتیجه پیش‌بینی")

                if 'impact' in predictions:
                    st.success(f"مقدار Impact Value پیش‌بینی‌شده: {predictions['impact']:.2f} J/m")
                    st.warning(
                        "توجه: مدل قادر به پیش‌بینی وضعیت 'عدم شکست' نیست و تنها یک مقدار عددی را پیش‌بینی می‌کند.")

                if 'tensile' in predictions:
                    st.success(f"مقدار Tensile Strength پیش‌بینی‌شده: {predictions['tensile']:.2f} MPa")
            except Exception as e:
                st.error(f"خطا در پیش‌بینی: {e}")

