import os
import time
import streamlit as st
from google.cloud import storage


def send_to_user_cloud(filename, target_url):

    if not os.path.exists(filename):
        st.error(f"File {filename} not found.")
        return

    if target_url.startswith("gs://"):
        try:
            bucket_name = target_url.replace("gs://", "").split("/")[0]
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_filename(filename)
            st.success(f"Uploaded to {target_url}{filename}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    elif target_url:
        time.sleep(1)
        st.success(f"Simulated upload to {target_url}{filename}")
