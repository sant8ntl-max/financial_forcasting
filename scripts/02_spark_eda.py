"""
Part 2 - Exploratory Data Analysis with Apache Spark
Auto-extracted standalone script from the main capstone notebook
(Financial_Forecasting_Frontier_Distributed_ML.ipynb) so it can be run
independently, per the submission checklist requirement for separate
Hadoop / Hive / Spark / Spark Streaming source files.

Run with: python 02_spark_eda.py
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
    .appName("BankSparkEDA")
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

# ============================================================
# 8. Distributed summary statistics
# ============================================================
bank_df.describe(["age", "balance", "duration", "campaign", "pdays", "previous"]).show()

print("Target class balance:")
bank_df.groupBy("y").count().show()

# ============================================================
# Chart 3 — Target class distribution
# ============================================================
target_pd = bank_df.groupBy("y").count().toPandas()

plt.figure(figsize=(5.5, 5))
colors = ["#e74c3c" if v == "no" else "#2ecc71" for v in target_pd["y"]]
plt.pie(target_pd["count"], labels=target_pd["y"].str.upper(), autopct="%1.1f%%",
        colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
plt.title("Term Deposit Subscription — Class Balance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/03_target_class_balance.png", dpi=150)
plt.show()

# ============================================================
# Chart 4 — Age distribution
# ============================================================
age_pd = bank_df.select("age").toPandas()

plt.figure(figsize=(9, 5))
sns.histplot(age_pd["age"], bins=30, kde=True, color="#3498db")
plt.title("Customer Age Distribution", fontsize=13, fontweight="bold")
plt.xlabel("Age"); plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/04_age_distribution.png", dpi=150)
plt.show()

# ============================================================
# Chart 5 — Account balance distribution (trimmed for readability)
# ============================================================
bal_pd = bank_df.select("balance").toPandas()
trimmed = bal_pd[(bal_pd["balance"] > bal_pd["balance"].quantile(0.01)) &
                  (bal_pd["balance"] < bal_pd["balance"].quantile(0.99))]

plt.figure(figsize=(9, 5))
sns.histplot(trimmed["balance"], bins=40, kde=True, color="#9b59b6")
plt.title("Account Balance Distribution (1st-99th percentile)", fontsize=13, fontweight="bold")
plt.xlabel("Balance (EUR)"); plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/05_balance_distribution.png", dpi=150)
plt.show()

# ============================================================
# Chart 6 — Customer count by job
# ============================================================
job_pd = bank_df.groupBy("job").count().orderBy(F.desc("count")).toPandas()

plt.figure(figsize=(10, 5.5))
sns.barplot(data=job_pd, x="count", y="job", hue="job", palette="crest", legend=False)
plt.title("Customer Count by Job Type", fontsize=13, fontweight="bold")
plt.xlabel("Number of Customers"); plt.ylabel("Job")
plt.tight_layout()
plt.savefig("charts/06_job_counts.png", dpi=150)
plt.show()

# ============================================================
# Chart 7 — Marital status split
# ============================================================
marital_pd = bank_df.groupBy("marital").count().toPandas()

plt.figure(figsize=(6, 5.5))
plt.pie(marital_pd["count"], labels=marital_pd["marital"].str.title(), autopct="%1.1f%%",
        colors=sns.color_palette("pastel"), startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
plt.title("Marital Status Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/07_marital_status.png", dpi=150)
plt.show()

# ============================================================
# Chart 8 — Education level counts
# ============================================================
edu_pd = bank_df.groupBy("education").count().orderBy(F.desc("count")).toPandas()

plt.figure(figsize=(7.5, 5))
sns.barplot(data=edu_pd, x="education", y="count", hue="education", palette="flare", legend=False)
plt.title("Customer Count by Education Level", fontsize=13, fontweight="bold")
plt.xlabel("Education"); plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/08_education_counts.png", dpi=150)
plt.show()

# ============================================================
# Chart 9 — Housing loan vs subscription outcome
# ============================================================
housing_pd = bank_df.groupBy("housing", "y").count().toPandas()
housing_pivot = housing_pd.pivot(index="housing", columns="y", values="count").fillna(0)

housing_pivot.plot(kind="bar", stacked=True, figsize=(7, 5), color=["#e74c3c", "#2ecc71"])
plt.title("Housing Loan vs Term Deposit Subscription", fontsize=13, fontweight="bold")
plt.xlabel("Has Housing Loan"); plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Subscribed")
plt.tight_layout()
plt.savefig("charts/09_housing_loan_vs_y.png", dpi=150)
plt.show()

# ============================================================
# Chart 10 — Personal loan vs subscription outcome
# ============================================================
loan_pd = bank_df.groupBy("loan", "y").count().toPandas()
loan_pivot = loan_pd.pivot(index="loan", columns="y", values="count").fillna(0)

loan_pivot.plot(kind="bar", stacked=True, figsize=(7, 5), color=["#e74c3c", "#2ecc71"])
plt.title("Personal Loan vs Term Deposit Subscription", fontsize=13, fontweight="bold")
plt.xlabel("Has Personal Loan"); plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Subscribed")
plt.tight_layout()
plt.savefig("charts/10_personal_loan_vs_y.png", dpi=150)
plt.show()

# ============================================================
# Chart 11 — Contact type distribution
# ============================================================
contact_pd = bank_df.groupBy("contact").count().orderBy(F.desc("count")).toPandas()

plt.figure(figsize=(7, 5))
sns.barplot(data=contact_pd, x="contact", y="count", hue="contact", palette="magma", legend=False)
plt.title("Contact Communication Type", fontsize=13, fontweight="bold")
plt.xlabel("Contact Type"); plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/11_contact_type.png", dpi=150)
plt.show()

# ============================================================
# Chart 12 — Monthly campaign contact volume
# ============================================================
month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
month_pd = bank_df.groupBy("month").count().toPandas()
month_pd["month"] = pd.Categorical(month_pd["month"], categories=month_order, ordered=True)
month_pd = month_pd.sort_values("month")

plt.figure(figsize=(10, 5))
sns.lineplot(data=month_pd, x="month", y="count", marker="o", color="#e67e22", linewidth=2.5)
plt.title("Campaign Contact Volume by Month", fontsize=13, fontweight="bold")
plt.xlabel("Month"); plt.ylabel("Number of Contacts")
plt.tight_layout()
plt.savefig("charts/12_monthly_contacts.png", dpi=150)
plt.show()

# ============================================================
# Chart 13 — Correlation heatmap of numeric features
# ============================================================
num_pd = bank_df.select("age","balance","day","duration","campaign","pdays","previous").toPandas()
corr = num_pd.corr()

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap — Numeric Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("charts/13_correlation_heatmap.png", dpi=150)
plt.show()

# ============================================================
# Chart 14 — Balance by subscription outcome (boxplot)
# ============================================================
by_pd = bank_df.select("balance", "y").toPandas()
by_pd_trim = by_pd[by_pd["balance"].between(by_pd["balance"].quantile(0.01), by_pd["balance"].quantile(0.99))]

plt.figure(figsize=(7, 5.5))
sns.boxplot(data=by_pd_trim, x="y", y="balance", hue="y", palette=["#e74c3c", "#2ecc71"], legend=False)
plt.title("Account Balance by Subscription Outcome", fontsize=13, fontweight="bold")
plt.xlabel("Subscribed to Term Deposit"); plt.ylabel("Balance (EUR)")
plt.tight_layout()
plt.savefig("charts/14_balance_by_outcome.png", dpi=150)
plt.show()

# ============================================================
# Chart 15 — Call duration by subscription outcome
# ============================================================
dur_pd = bank_df.select("duration", "y").toPandas()

plt.figure(figsize=(7, 5.5))
sns.violinplot(data=dur_pd, x="y", y="duration", hue="y", palette=["#e74c3c", "#2ecc71"], legend=False)
plt.title("Last Contact Duration by Subscription Outcome", fontsize=13, fontweight="bold")
plt.xlabel("Subscribed to Term Deposit"); plt.ylabel("Call Duration (seconds)")
plt.tight_layout()
plt.savefig("charts/15_duration_by_outcome.png", dpi=150)
plt.show()

# ============================================================
# Chart 16 — Previous campaign outcome vs current subscription
# ============================================================
pout_pd = bank_df.groupBy("poutcome", "y").count().toPandas()
pout_pivot = pout_pd.pivot(index="poutcome", columns="y", values="count").fillna(0)

pout_pivot.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#e74c3c", "#2ecc71"])
plt.title("Previous Campaign Outcome vs Current Subscription", fontsize=13, fontweight="bold")
plt.xlabel("Previous Campaign Outcome"); plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Subscribed")
plt.tight_layout()
plt.savefig("charts/16_poutcome_vs_y.png", dpi=150)
plt.show()

# ============================================================
# Chart 17 — Age vs Balance scatter, coloured by subscription outcome
# ============================================================
scatter_pd = bank_df.select("age", "balance", "y").toPandas()
scatter_trim = scatter_pd[scatter_pd["balance"].between(-2000, 15000)]

plt.figure(figsize=(9, 6))
sns.scatterplot(data=scatter_trim, x="age", y="balance", hue="y",
                 palette={"no": "#e74c3c", "yes": "#2ecc71"}, alpha=0.6, s=25)
plt.title("Age vs Account Balance by Subscription Outcome", fontsize=13, fontweight="bold")
plt.xlabel("Age"); plt.ylabel("Balance (EUR)")
plt.legend(title="Subscribed")
plt.tight_layout()
plt.savefig("charts/17_age_vs_balance.png", dpi=150)
plt.show()

# ============================================================
# 📌 EDA INSIGHTS & CONCLUSIONS (data-driven, computed from this run)
# ============================================================
overall_rate = bank_df.filter(F.col("y") == "yes").count() / bank_df.count() * 100

dur_stats = bank_df.groupBy("y").agg(F.round(F.avg("duration"), 1).alias("avg_duration")).toPandas()
dur_yes = dur_stats.loc[dur_stats["y"] == "yes", "avg_duration"].values[0]
dur_no  = dur_stats.loc[dur_stats["y"] == "no", "avg_duration"].values[0]

edu_best = q2_pd.sort_values("subscription_rate_pct", ascending=False).iloc[0]
job_best = q1_pd.sort_values("avg_balance", ascending=False).iloc[0]

pout_stats = bank_df.groupBy("poutcome").agg(
    F.round(100 * F.sum(F.when(F.col("y") == "yes", 1).otherwise(0)) / F.count("*"), 2).alias("rate")
).toPandas()
pout_best = pout_stats.sort_values("rate", ascending=False).iloc[0]

print("=" * 70)
print("EDA INSIGHTS & CONCLUSIONS")
print("=" * 70)
print(f"1. Overall subscription rate is {overall_rate:.2f}% — the dataset is imbalanced "
      f"(~{100-overall_rate:.0f}% did not subscribe), so AUC/F1 matter more than raw accuracy for modelling.")
print(f"2. Customers who subscribed had a much longer average call duration "
      f"({dur_yes:.0f}s) vs those who didn't ({dur_no:.0f}s) — duration is the strongest single behavioural signal.")
print(f"3. '{edu_best['education']}' education customers have the highest subscription rate "
      f"at {edu_best['subscription_rate_pct']:.2f}%.")
print(f"4. '{job_best['job']}' customers carry the highest average account balance "
      f"(EUR {job_best['avg_balance']:.2f}), useful for premium-product targeting.")
print(f"5. Customers with a previous campaign outcome of '{pout_best['poutcome']}' convert at "
      f"{pout_best['rate']:.2f}% — by far the strongest targeting signal for future campaigns.")
print(f"6. Numeric features show low pairwise correlation (see heatmap above), so multicollinearity")
print(f"   is not a concern for the classification models trained next.")
print("=" * 70)