import time
import pandas as pd
from spark_utils import get_spark


def run_scalability_test(file_path):
    
    results = []

    for n in [1, 2, 4, 8]:
        start = time.time()
        spark = get_spark(n)
        spark.read.csv(file_path).count()
        spark.stop()

        results.append({
            "Nodes": n,
            "Time (s)": time.time() - start
        })

    df = pd.DataFrame(results)
    t1 = df.iloc[0]["Time (s)"]

    df["Speedup"] = t1 / df["Time (s)"]
    df["Efficiency (%)"] = (df["Speedup"] / df["Nodes"]) * 100

    return df
