import pandas as pd
import pyspark.sql.functions as F


def run_descriptive_stats(df):

    row_count = df.count()
    col_count = len(df.columns)

    null_count = (
        df.select([
            F.count(F.when(F.col(c).isNull(), c)).alias(c)
            for c in df.columns
        ])
        .toPandas()
        .values
        .sum()
    )

    unique_val = df.select(df.columns[0]).distinct().count()

    return pd.DataFrame({
        "Metric": ["Rows", "Columns", "Null Values", "Unique (First Column)"],
        "Value": [row_count, col_count, int(null_count), unique_val]
    })
