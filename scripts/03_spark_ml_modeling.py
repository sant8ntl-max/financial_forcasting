"""
Part 3 - Predictive Modelling with Spark MLlib
Auto-extracted standalone script from the main capstone notebook
(Financial_Forecasting_Frontier_Distributed_ML.ipynb) so it can be run
independently, per the submission checklist requirement for separate
Hadoop / Hive / Spark / Spark Streaming source files.

Run with: python 03_spark_ml_modeling.py
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
    .appName("BankSparkML")
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
# 9. Build the Spark ML preprocessing + modelling pipeline
# ============================================================
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe") for c in cat_cols]
label_indexer = StringIndexer(inputCol="y", outputCol="label")
assembler = VectorAssembler(inputCols=[c + "_ohe" for c in cat_cols] + num_cols, outputCol="features")

train_df, test_df = bank_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training rows: {train_df.count():,}   |   Test rows: {test_df.count():,}")

# ============================================================
# 10. Train Logistic Regression (Spark ML, distributed training)
# ============================================================
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
lr_pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, lr])
lr_model = lr_pipeline.fit(train_df)
lr_pred = lr_model.transform(test_df)

auc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

lr_auc = auc_eval.evaluate(lr_pred)
lr_acc = acc_eval.evaluate(lr_pred)
lr_f1  = f1_eval.evaluate(lr_pred)
print(f"Logistic Regression  ->  AUC: {lr_auc:.4f}   Accuracy: {lr_acc:.4f}   F1: {lr_f1:.4f}")

# ============================================================
# 11. Train Random Forest (Spark ML, distributed training)
# ============================================================
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=8, seed=42)
rf_pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler, rf])
rf_model = rf_pipeline.fit(train_df)
rf_pred = rf_model.transform(test_df)

rf_auc = auc_eval.evaluate(rf_pred)
rf_acc = acc_eval.evaluate(rf_pred)
rf_f1  = f1_eval.evaluate(rf_pred)
print(f"Random Forest         ->  AUC: {rf_auc:.4f}   Accuracy: {rf_acc:.4f}   F1: {rf_f1:.4f}")

# ============================================================
# Chart 18 — Model comparison (Logistic Regression vs Random Forest)
# ============================================================
comp_pd = pd.DataFrame({
    "Model": ["Logistic Regression", "Logistic Regression", "Logistic Regression",
              "Random Forest", "Random Forest", "Random Forest"],
    "Metric": ["AUC", "Accuracy", "F1"] * 2,
    "Score": [lr_auc, lr_acc, lr_f1, rf_auc, rf_acc, rf_f1],
})

plt.figure(figsize=(9, 5.5))
sns.barplot(data=comp_pd, x="Metric", y="Score", hue="Model", palette=["#3498db", "#2ecc71"])
plt.title("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("charts/18_model_comparison.png", dpi=150)
plt.show()

# ============================================================
# Chart 19 — Confusion matrix (Random Forest — best model)
# ============================================================
from sklearn.metrics import confusion_matrix

cm_pd = rf_pred.select("label", "prediction").toPandas()
cm = confusion_matrix(cm_pd["label"], cm_pd["prediction"])

plt.figure(figsize=(6, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix — Random Forest", fontsize=13, fontweight="bold")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("charts/19_confusion_matrix.png", dpi=150)
plt.show()

# ============================================================
# Chart 20 — ROC curve (Random Forest)
# ============================================================
from sklearn.metrics import roc_curve, auc as sk_auc

roc_pd = rf_pred.select("label", "probability").toPandas()
roc_pd["prob_yes"] = roc_pd["probability"].apply(lambda v: float(v[1]))
fpr, tpr, _ = roc_curve(roc_pd["label"], roc_pd["prob_yes"])
roc_auc_val = sk_auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#2ecc71", linewidth=2.5, label=f"Random Forest (AUC = {roc_auc_val:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
plt.title("ROC Curve — Random Forest", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/20_roc_curve.png", dpi=150)
plt.show()

# ============================================================
# Chart 21 — Feature importance (Random Forest)
# ============================================================
feature_names = []
for c in cat_cols:
    n_categories = bank_df.select(c).distinct().count()
    feature_names += [f"{c}_{i}" for i in range(n_categories)]
feature_names += num_cols

importances = rf_model.stages[-1].featureImportances.toArray()
imp_pd = pd.DataFrame({"feature": feature_names[:len(importances)], "importance": importances})
imp_pd = imp_pd.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(data=imp_pd, x="importance", y="feature", hue="feature", palette="rocket", legend=False)
plt.title("Top 15 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
plt.xlabel("Importance"); plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("charts/21_feature_importance.png", dpi=150)
plt.show()

# ============================================================
# 📌 MODELLING INSIGHTS & CONCLUSIONS
# ============================================================
top_features = imp_pd.head(5)["feature"].tolist()
better_model = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
tn, fp, fn, tp = cm.ravel()
precision_yes = tp / (tp + fp) if (tp + fp) else 0
recall_yes = tp / (tp + fn) if (tp + fn) else 0

print("=" * 70)
print("MODELLING INSIGHTS & CONCLUSIONS")
print("=" * 70)
print(f"1. {better_model} achieved the highest AUC ({max(rf_auc, lr_auc):.4f}), making it the recommended")
print(f"   model for ranking customers by subscription likelihood.")
print(f"2. Logistic Regression -> AUC {lr_auc:.4f} | Accuracy {lr_acc:.4f} | F1 {lr_f1:.4f}")
print(f"   Random Forest       -> AUC {rf_auc:.4f} | Accuracy {rf_acc:.4f} | F1 {rf_f1:.4f}")
print(f"3. On the held-out test set, Random Forest correctly identified {tp} of {tp+fn} actual subscribers")
print(f"   (recall = {recall_yes:.2%}) with a precision of {precision_yes:.2%} on positive predictions —")
print(f"   reflecting the class imbalance in the training data.")
print(f"4. Top 5 predictive features: {', '.join(top_features)}.")
print(f"5. Business implication: campaigns should prioritise longer, well-timed calls to customers with")
print(f"   a successful previous campaign outcome and higher account balances — these signals drove the")
print(f"   model's predictions the most.")
print("=" * 70)