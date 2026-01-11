# Cloud-Based Distributed Data Processing Service

This project was developed as part of a university course in Cloud Computing
and Distributed Systems.

The link on drive: 
	https://colab.research.google.com/drive/1bkoiWzDtpHtG9beyraeYjGNeFASG266w?usp=sharing

## 📌 Project Description
The system allows users to upload datasets and perform distributed data
analytics using Apache Spark and PySpark. The application runs on a cloud
development environment and provides descriptive statistics, machine learning
analysis, and scalability benchmarking.

## 🛠 Technologies Used
- Python
- Apache Spark (PySpark)
- Streamlit
- Pandas


## 🚀 Features
- Upload CSV / JSON datasets
- Automatic schema inference
- Descriptive statistics
- Machine learning models:
  - Linear Regression
  - KMeans
  - Random Forest
  - Decision Tree
- Scalability analysis with Speedup and Efficiency
- Download results as CSV files

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
