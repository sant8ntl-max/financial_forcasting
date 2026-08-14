"""
Part 5 - Data Parallelism Benchmarking
Auto-extracted standalone script from the main capstone notebook
(Financial_Forecasting_Frontier_Distributed_ML.ipynb) so it can be run
independently, per the submission checklist requirement for separate
Hadoop / Hive / Spark / Spark Streaming source files.

Run with: python 05_data_parallelism_benchmark.py
Requires: pip install pyspark pyarrow pandas matplotlib seaborn scikit-learn
Also requires Java 11+ (PySpark's runtime dependency) — install with:
  Ubuntu/Debian: sudo apt-get install openjdk-17-jdk-headless
  Mac (brew):    brew install openjdk@17
  Windows:       install Adoptium Temurin 17 and set JAVA_HOME
"""
import os, time, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

sns.set_style("whitegrid")
os.makedirs("charts", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("BankDataParallelism")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("marital", StringType(), True),
    StructField("education", StringType(), True),
    StructField("default", StringType(), True),
    StructField("balance", IntegerType(), True),
    StructField("housing", StringType(), True),
    StructField("loan", StringType(), True),
    StructField("contact", StringType(), True),
    StructField("day", IntegerType(), True),
    StructField("month", StringType(), True),
    StructField("duration", IntegerType(), True),
    StructField("campaign", IntegerType(), True),
    StructField("pdays", IntegerType(), True),
    StructField("previous", IntegerType(), True),
    StructField("poutcome", StringType(), True),
    StructField("y", StringType(), True),
])

DATA_PATH = os.environ.get("BANK_CSV_PATH", "../data/bank.csv")
bank_df = spark.read.csv(DATA_PATH, header=True, schema=schema).repartition(8)
bank_df.createOrReplaceTempView("bank_data")
print(f"Loaded {bank_df.count()} rows from {DATA_PATH}")

# ---------------------------------------------------------------------------
# ============================================================
# 15. Benchmark: execution time vs number of partitions
# ============================================================
partition_counts = [1, 2, 4, 8, 16]
timings = []

for n in partition_counts:
    repartitioned = bank_df.repartition(n)
    start = time.time()
    repartitioned.groupBy("job", "education").agg(
        F.avg("balance"), F.avg("duration"), F.count("*")
    ).collect()
    elapsed = time.time() - start
    timings.append({"partitions": n, "time_sec": elapsed})
    print(f"Partitions: {n:>2}  ->  Time: {elapsed:.3f}s")

timing_pd = pd.DataFrame(timings)

# ============================================================
# Chart 24 — Execution time vs partition count
# ============================================================
plt.figure(figsize=(8.5, 5.5))
sns.barplot(data=timing_pd, x="partitions", y="time_sec", hue="partitions",
            palette="viridis", legend=False)
plt.title("Query Execution Time vs Degree of Data Parallelism", fontsize=13, fontweight="bold")
plt.xlabel("Number of Partitions"); plt.ylabel("Execution Time (seconds)")
plt.tight_layout()
plt.savefig("charts/24_parallelism_benchmark.png", dpi=150)
plt.show()

print("\nNote: on this small ~4.5K row demo dataset, scheduling/shuffle overhead can dominate at very high "
      "partition counts. On production-scale banking data (millions/billions of rows), increasing partitions "
      "up to the cluster's core count yields near-linear speedups.")

# ============================================================
# 16. Clean shutdown
# ============================================================
spark.catalog.dropTempView("bank_data")
print("Notebook run complete. Spark session remains active for further exploration; call spark.stop() when done.")