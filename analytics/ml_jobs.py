import pandas as pd
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import pyspark.sql.functions as F


def run_ml_jobs(df, numeric_cols):

    assembler = VectorAssembler(
        inputCols=numeric_cols[:-1],
        outputCol="features",
        handleInvalid="skip"
    )

    ml_data = assembler.transform(df).select(
        "features", F.col(numeric_cols[-1]).alias("label")
    )

    train, test = ml_data.randomSplit([0.8, 0.2], seed=42)

    lr_rmse = RegressionEvaluator().evaluate(
        LinearRegression().fit(train).transform(test)
    )

    km_cost = KMeans(k=3, seed=1).fit(ml_data).summary.trainingCost

    indexer = StringIndexer(inputCol="label", outputCol="idx").fit(ml_data)
    c_data = indexer.transform(ml_data)
    c_train, c_test = c_data.randomSplit([0.8, 0.2])

    evaluator = MulticlassClassificationEvaluator(
        labelCol="idx", metricName="accuracy"
    )

    rf_acc = evaluator.evaluate(
        RandomForestClassifier(labelCol="idx")
        .fit(c_train).transform(c_test)
    )

    dt_acc = evaluator.evaluate(
        DecisionTreeClassifier(labelCol="idx")
        .fit(c_train).transform(c_test)
    )

    return pd.DataFrame({
        "Algorithm": ["Linear Regression", "KMeans", "Random Forest", "Decision Tree"],
        "Metric": ["RMSE", "WSSSE", "Accuracy", "Accuracy"],
        "Result": [
            f"{lr_rmse:.4f}",
            f"{km_cost:.2f}",
            f"{rf_acc*100:.2f}%",
            f"{dt_acc*100:.2f}%"
        ]
    })
