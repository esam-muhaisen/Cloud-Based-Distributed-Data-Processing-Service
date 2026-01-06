
import streamlit as st
import os


def download_csv_button(label, file_path):
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label=label,
                data=f,
                file_name=file_path,
                mime="text/csv"
            )
