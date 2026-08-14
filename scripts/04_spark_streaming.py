"""
Part 4 - Real-Time Transaction Processing with Spark Structured Streaming
Auto-extracted standalone script from the main capstone notebook
(Financial_Forecasting_Frontier_Distributed_ML.ipynb) so it can be run
independently, per the submission checklist requirement for separate
Hadoop / Hive / Spark / Spark Streaming source files.

Run with: python 04_spark_streaming.py
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
    .appName("BankSparkStreaming")
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
# 12. Set up a streaming source directory + streaming query
# ============================================================
stream_dir = "streaming_input"
shutil.rmtree(stream_dir, ignore_errors=True)
os.makedirs(stream_dir, exist_ok=True)

stream_df = spark.readStream.csv(stream_dir, header=True, schema=schema)

live_agg = (
    stream_df.groupBy("job")
    .agg(F.count("*").alias("txn_count"), F.round(F.avg("balance"), 2).alias("avg_balance"))
)

query = (
    live_agg.writeStream
    .outputMode("complete")
    .format("memory")
    .queryName("live_transactions")
    .start()
)
print("Streaming query started:", query.name, "| active:", query.isActive)

# ============================================================
# 13. Simulate incoming real-time transaction batches
# ============================================================
full_pd = bank_df.toPandas()
batch_size = 400
n_batches = 5
batch_history = []

for i in range(n_batches):
    chunk = full_pd.sample(n=batch_size, random_state=i)
    chunk.to_csv(f"{stream_dir}/batch_{i}.csv", index=False)
    time.sleep(1)
    query.processAllAvailable()

    snapshot = spark.sql("SELECT SUM(txn_count) as total_txns, ROUND(AVG(avg_balance),2) as overall_avg_balance "
                          "FROM live_transactions").toPandas()
    snapshot["batch"] = i + 1
    batch_history.append(snapshot)
    print(f"Batch {i+1}/{n_batches} processed  ->  cumulative transactions seen: "
          f"{int(snapshot['total_txns'].iloc[0])}")

print("\nFinal live aggregation snapshot (by job):")
spark.sql("SELECT * FROM live_transactions ORDER BY txn_count DESC").show(12, truncate=False)

# ============================================================
# Chart 22 — Live streaming: transactions processed per job (final snapshot)
# ============================================================
live_pd = spark.sql("SELECT * FROM live_transactions ORDER BY txn_count DESC").toPandas()

plt.figure(figsize=(9, 5.5))
sns.barplot(data=live_pd, x="txn_count", y="job", hue="job", palette="cubehelix", legend=False)
plt.title("Live Streaming Aggregation — Transactions Processed by Job (Final Snapshot)",
          fontsize=12.5, fontweight="bold")
plt.xlabel("Transaction Count"); plt.ylabel("Job")
plt.tight_layout()
plt.savefig("charts/22_streaming_by_job.png", dpi=150)
plt.show()

# ============================================================
# Chart 23 — Cumulative transactions processed over streaming batches
# ============================================================
history_pd = pd.concat(batch_history, ignore_index=True)

plt.figure(figsize=(9, 5))
sns.lineplot(data=history_pd, x="batch", y="total_txns", marker="o", linewidth=2.5, color="#e67e22")
plt.title("Cumulative Transactions Processed Across Streaming Batches", fontsize=12.5, fontweight="bold")
plt.xlabel("Batch Number"); plt.ylabel("Cumulative Transaction Count")
plt.xticks(history_pd["batch"].unique())
plt.tight_layout()
plt.savefig("charts/23_streaming_cumulative.png", dpi=150)
plt.show()

# ============================================================
# 📌 STREAMING INSIGHTS & CONCLUSIONS
# ============================================================
top_streaming_job = live_pd.sort_values("txn_count", ascending=False).iloc[0]
total_streamed = int(history_pd["total_txns"].max())

print("=" * 70)
print("STREAMING INSIGHTS & CONCLUSIONS")
print("=" * 70)
print(f"1. {total_streamed} simulated transactions were processed live across {n_batches} streaming batches,")
print(f"   with the aggregation table updating automatically after every new file arrival.")
print(f"2. The '{top_streaming_job['job']}' segment generated the highest transaction volume in the live feed")
print(f"   ({int(top_streaming_job['txn_count'])} transactions, avg balance EUR {top_streaming_job['avg_balance']:.2f}).")
print(f"3. This demonstrates Spark Structured Streaming's core value for banking: the same aggregation logic")
print(f"   used in batch EDA (Part 2) runs unchanged on a continuous stream, enabling real-time fraud")
print(f"   monitoring and live dashboards without separate batch/streaming codebases.")
print("=" * 70)

# ============================================================
# 14. Stop the streaming query cleanly
# ============================================================
query.stop()
print("Streaming query stopped. active:", query.isActive)