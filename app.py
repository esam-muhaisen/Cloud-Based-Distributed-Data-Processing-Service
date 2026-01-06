# app.py

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


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("⚙️ User Configuration")
    user_storage_link = st.text_input(
        "Enter Destination URL (GCS / S3 / Azure)",
        placeholder="gs://your-bucket-name/"
    )

uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv", "json", "txt"])

if uploaded_file:
    file_path = f"data.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    spark = get_spark(4)

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    num_cols = [c for c, t in df.dtypes if t in ["int", "double", "float"]]
    df_numeric = df.select(num_cols).dropna()

    st.header("📊 Descriptive Statistics")
    stats_df = run_descriptive_stats(df)
    st.table(stats_df)
    stats_df.to_csv(STATS_PATH, index=False)
    download_csv_button("Download Statistics", STATS_PATH)

    if st.button("Run ML Jobs") and len(num_cols) > 1:
        st.header("🤖 Machine Learning")
        ml_df = run_ml_jobs(df_numeric, num_cols)
        st.table(ml_df)
        ml_df.to_csv(ML_RESULTS_PATH, index=False)

    if st.button("Run Scalability Test"):
        st.header("⚡ Scalability Analysis")
        sc_df = run_scalability_test(file_path)
        st.table(sc_df)
