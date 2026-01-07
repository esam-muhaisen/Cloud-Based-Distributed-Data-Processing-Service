# # app.py

# import streamlit as st
# import pandas as pd
# import pyspark.sql.functions as F

# from config import *
# from spark_utils import get_spark
# from cloud_utils import send_to_user_cloud
# from file_utils import download_csv_button

# from analytics.descriptive import run_descriptive_stats
# from analytics.ml_jobs import run_ml_jobs
# from analytics.scalability import run_scalability_test


# st.set_page_config(page_title=APP_TITLE, layout="wide")
# st.title(APP_TITLE)

# with st.sidebar:
#     st.header("User Configuration")
#     user_storage_link = st.text_input(
#         "Enter Destination URL (GCS / S3 / Azure)",
#         placeholder="gs://your-bucket-name/"
#     )

# uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "json", "txt"])

# if uploaded_file:
#     file_path = f"data.{uploaded_file.name.split('.')[-1]}"
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     spark = get_spark(4)

#     df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
#     num_cols = [c for c, t in df.dtypes if t in ["int", "double", "float"]]
#     df_numeric = df.select(num_cols).dropna()

#     st.header("Descriptive Statistics")
#     stats_df = run_descriptive_stats(df)
#     st.table(stats_df)
#     stats_df.to_csv(STATS_PATH, index=False)
#     download_csv_button("Download Statistics", STATS_PATH)

#     if st.button("Run ML Jobs") and len(num_cols) > 1:
#         st.header("Machine Learning")
#         ml_df = run_ml_jobs(df_numeric, num_cols)
#         st.table(ml_df)
#         ml_df.to_csv(ML_RESULTS_PATH, index=False)

#     if st.button("Run Scalability Test"):
#         st.header("Scalability Analysis")
#         sc_df = run_scalability_test(file_path)
#         st.table(sc_df)

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

# --- تهيئة الـ Session State للحفاظ على حالة الأزرار ---
if 'ml_run_clicked' not in st.session_state:
    st.session_state.ml_run_clicked = False

if 'scalability_run_clicked' not in st.session_state:
    st.session_state.scalability_run_clicked = False

# --- التخزين المؤقت للعمليات الثقيلة (Caching) ---
@st.cache_data
def cached_ml_jobs(data, columns):
    return run_ml_jobs(data, columns)

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
    # زر لإعادة ضبط التطبيق ومسح النتائج
    if st.button("Clear Results"):
        st.session_state.ml_run_clicked = False
        st.session_state.scalability_run_clicked = False
        st.rerun()

# --- رفع الملفات ---
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "json", "txt"])

if uploaded_file:
    file_path = f"data.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # تشغيل Spark (يفضل وضعه داخل cache أو إدارته بحذر)
    spark = get_spark(4)
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    
    num_cols = [c for c, t in df.dtypes if t in ["int", "double", "float"]]
    df_numeric = df.select(num_cols).dropna()

    # --- القسم الأول: الإحصاء الوصفي (يظهر دائماً عند رفع ملف) ---
    st.header("Descriptive Statistics")
    stats_df = run_descriptive_stats(df)
    st.table(stats_df)
    stats_df.to_csv(STATS_PATH, index=False)
    download_csv_button("Download Statistics", STATS_PATH)

    st.divider() # خط فاصل للتنظيم

    # --- القسم الثاني: Machine Learning ---
    col1, col2 = st.columns(2) # وضع الأزرار بجانب بعضها لسهولة الاستخدام

    with col1:
        if st.button("Run ML Jobs") and len(num_cols) > 1:
            st.session_state.ml_run_clicked = True

    if st.session_state.ml_run_clicked:
        st.subheader("Machine Learning Results")
        with st.spinner("Running ML Models..."):
            ml_df = cached_ml_jobs(df_numeric, num_cols)
            st.table(ml_df)
            ml_df.to_csv(ML_RESULTS_PATH, index=False)

    # --- القسم الثالث: Scalability Test ---
    with col2:
        if st.button("Run Scalability Test"):
            st.session_state.scalability_run_clicked = True

    if st.session_state.scalability_run_clicked:
        st.subheader("Scalability Analysis Results")
        with st.spinner("Testing Scalability..."):
            sc_df = cached_scalability_test(file_path)
            st.table(sc_df)