"""
Part 1 - Hadoop (HDFS-style) Storage & Hive-style Querying
Auto-extracted standalone script from the main capstone notebook
(Financial_Forecasting_Frontier_Distributed_ML.ipynb) so it can be run
independently, per the submission checklist requirement for separate
Hadoop / Hive / Spark / Spark Streaming source files.

Run with: python 01_hadoop_hive_storage_and_querying.py
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
    .appName("BankHadoopHive")
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
# 3. Load raw data into a distributed DataFrame (HDFS-style ingestion)
# ============================================================
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

bank_df = spark.read.csv(DATA_PATH, header=True, schema=schema)
bank_df = bank_df.repartition(8)   # simulate data spread across 8 HDFS blocks/executors

print(f"Total records ingested: {bank_df.count():,}")
print(f"Number of partitions (simulated HDFS blocks): {bank_df.rdd.getNumPartitions()}")
bank_df.printSchema()

# ============================================================
# 4. Register as a Hive-style managed table + temp view, run HiveQL
# ============================================================
bank_df.createOrReplaceTempView("bank_data")
bank_df.write.mode("overwrite").saveAsTable("bank_managed")

print("Tables/views available in the Hive-style warehouse:")
spark.sql("SHOW TABLES").show(truncate=False)

bank_df.show(5, truncate=False)

# ============================================================
# 5. HiveQL-style analytical queries
# ============================================================
print("Q1. Average balance & subscription count by job role")
q1 = spark.sql("""
    SELECT job,
           ROUND(AVG(balance), 2)                         AS avg_balance,
           COUNT(*)                                        AS customers,
           SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)      AS subscribers
    FROM bank_data
    GROUP BY job
    ORDER BY avg_balance DESC
""")
q1.show(12, truncate=False)

print("Q2. Subscription rate (%) by education level")
q2 = spark.sql("""
    SELECT education,
           ROUND(100 * SUM(CASE WHEN y='yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS subscription_rate_pct,
           COUNT(*) AS customers
    FROM bank_data
    GROUP BY education
    ORDER BY subscription_rate_pct DESC
""")
q2.show(truncate=False)

print("Q3. Campaign contact volume by month")
q3 = spark.sql("""
    SELECT month, COUNT(*) AS contacts, ROUND(AVG(duration),1) AS avg_call_duration
    FROM bank_data
    GROUP BY month
    ORDER BY contacts DESC
""")
q3.show(12, truncate=False)

# ============================================================
# 6. Chart 1 — Average balance by job (from Hive-style query q1)
# ============================================================
q1_pd = q1.toPandas()

plt.figure(figsize=(10, 5))
sns.barplot(data=q1_pd, x="avg_balance", y="job", hue="job", palette="viridis", legend=False)
plt.title("Average Account Balance by Job Role (Hive-style Query)", fontsize=13, fontweight="bold")
plt.xlabel("Average Balance (EUR)"); plt.ylabel("Job")
plt.tight_layout()
plt.savefig("charts/01_avg_balance_by_job.png", dpi=150)
plt.show()

# ============================================================
# 7. Chart 2 — Subscription rate by education level
# ============================================================
q2_pd = q2.toPandas()

plt.figure(figsize=(7, 5))
sns.barplot(data=q2_pd, x="education", y="subscription_rate_pct", hue="education",
            palette="mako", legend=False)
plt.title("Term Deposit Subscription Rate by Education Level", fontsize=13, fontweight="bold")
plt.xlabel("Education"); plt.ylabel("Subscription Rate (%)")
plt.tight_layout()
plt.savefig("charts/02_subscription_rate_by_education.png", dpi=150)
plt.show()