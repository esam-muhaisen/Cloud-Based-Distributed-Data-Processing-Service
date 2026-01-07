import streamlit as st
import pandas as pd
import pyspark.sql.functions as F

from config import *
from spark_utils import get_spark
from cloud_utils import send_to_user_cloud
from file_utils import download_csv_button

from analytics.descriptive import run_descriptive_stats
from analytics.ml_jobs import run_ml_jobs
from analytics.scalability import run_scalability_test

# إعداد الصفحة
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# --- تهيئة الـ Session State للحفاظ على الحالة والبيانات ---
if 'ml_run_clicked' not in st.session_state:
    st.session_state.ml_run_clicked = False
if 'scalability_run_clicked' not in st.session_state:
    st.session_state.scalability_run_clicked = False
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'sc_results' not in st.session_state:
    st.session_state.sc_results = None

# --- التخزين المؤقت للعمليات الثقيلة (Caching) ---
@st.cache_data
def cached_ml_jobs(_data, columns):
    return run_ml_jobs(_data, columns)

@st.cache_data
def cached_scalability_test(path):
    return run_scalability_test(path)

# --- واجهة المستخدم (Sidebar) ---
with st.sidebar:
    st.header("User Configuration")
    user_storage_link = st.text_input(
        "Enter Destination URL (GCS / S3 / Azure)",
        placeholder="gs://your-bucket-name/"
    )
    if st.button("Clear All Results"):
        st.session_state.ml_run_clicked = False
        st.session_state.scalability_run_clicked = False
        st.session_state.ml_results = None
        st.session_state.sc_results = None
        st.rerun()

# --- رفع الملفات ---
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "json", "txt"])

if uploaded_file:
    file_path = f"data.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # إعداد Spark
    spark = get_spark(4)
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    
    num_cols = [c for c, t in df.dtypes if t in ["int", "double", "float"]]
    df_numeric = df.select(num_cols).dropna()

    # --- 1. Descriptive Statistics ---
    st.header("1. Descriptive Statistics")
    stats_df = run_descriptive_stats(df)
    st.table(stats_df)
    stats_df.to_csv(STATS_PATH, index=False)
    download_csv_button("Download Descriptive Statistics", STATS_PATH)

    st.divider()

    # --- الأزرار الجانبية للتشغيل ---
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Run ML Jobs") and len(num_cols) > 1:
            st.session_state.ml_run_clicked = True
            with st.spinner("Running ML Models..."):
                st.session_state.ml_results = cached_ml_jobs(df_numeric, num_cols)

    with col_btn2:
        if st.button("Run Scalability Test"):
            st.session_state.scalability_run_clicked = True
            with st.spinner("Testing Scalability..."):
                st.session_state.sc_results = cached_scalability_test(file_path)

    # --- عرض النتائج وأزرار التحميل في الأسفل ---
    
    # عرض نتائج Machine Learning
    if st.session_state.ml_run_clicked and st.session_state.ml_results is not None:
        st.header("2. Machine Learning Results")
        st.table(st.session_state.ml_results)
        # حفظ الملف محلياً لتوفير زر التحميل
        st.session_state.ml_results.to_csv(ML_RESULTS_PATH, index=False)
        download_csv_button("Download ML Results", ML_RESULTS_PATH)

    # عرض نتائج Scalability
    if st.session_state.scalability_run_clicked and st.session_state.sc_results is not None:
        st.header("3. Scalability Analysis Results")
        st.table(st.session_state.sc_results)
        # حفظ الملف محلياً لتوفير زر التحميل
        sc_path = "scalability_results.csv"
        st.session_state.sc_results.to_csv(sc_path, index=False)
        download_csv_button("Download Scalability Results", sc_path)