# 🏦 Financial Forecasting Frontier — Distributed ML for Banking Analytics

**AlmaBetter Masters in Data Science — Capstone Project**

An end-to-end distributed data pipeline built on a real-world bank telemarketing dataset, simulating how a modern
banking data platform stores, queries, explores, models, and streams customer data using distributed computing
tools.

![banner](charts/00_banner.png)

## 🎯 Objectives

| # | Objective | Tech Used |
|---|-----------|-----------|
| 1 | Data storage & querying | Hadoop (HDFS concepts) + Hive (simulated with Spark SQL) |
| 2 | Exploratory Data Analysis | Apache Spark |
| 3 | Predictive modelling | Spark MLlib (Logistic Regression, Random Forest) |
| 4 | Real-time transaction processing | Spark Structured Streaming |
| 5 | Data parallelism benchmarking | Spark partitioning |

## 📂 Repository Structure (mapped to submission checklist)

| Checklist Requirement | Where to find it |
|---|---|
| 1. Code + output for all 5 parts | `notebooks/Financial_Forecasting_Frontier_Distributed_ML.ipynb` (single Colab notebook, all outputs saved) |
| 2. Separate source scripts for Hadoop, Hive, Spark, Spark Streaming | `scripts/01_hadoop_hive_storage_and_querying.py`, `02_spark_eda.py`, `03_spark_ml_modeling.py`, `04_spark_streaming.py`, `05_data_parallelism_benchmark.py` — each independently runnable |
| 3. Sample data chunks for Spark Streaming | `streaming_sample_data/chunk_0.csv` … `chunk_4.csv` (300 rows each, the exact chunks used to simulate the live feed) |
| 4 & 5. Video presentation (15+ min) | Submitted separately under "Video link"; script used to record it: `docs/Video_Presentation_Script.docx` |
| 6. Reflective summary (challenges + learnings) | `docs/Reflective_Summary.docx` |

Full folder layout:

```
├── notebooks/
│   └── Financial_Forecasting_Frontier_Distributed_ML.ipynb   # Main Colab notebook (run this) — all outputs saved
├── scripts/
│   ├── 01_hadoop_hive_storage_and_querying.py                 # Standalone — Part 1
│   ├── 02_spark_eda.py                                         # Standalone — Part 2
│   ├── 03_spark_ml_modeling.py                                 # Standalone — Part 3
│   ├── 04_spark_streaming.py                                   # Standalone — Part 4
│   ├── 05_data_parallelism_benchmark.py                        # Standalone — Part 5
│   └── bank_distributed_ml.py                                  # Full notebook exported as one script
├── streaming_sample_data/
│   └── chunk_0.csv … chunk_4.csv                                # Sample transaction chunks used for streaming demo
├── data/
│   └── bank.csv                                                # Bank Marketing dataset (4,521 rows)
├── charts/
│   └── *.png                                                   # All 25 generated charts/figures
├── docs/
│   ├── Financial_Forecasting_Frontier_Report.docx              # Full written project report
│   ├── Reflective_Summary.docx                                 # Challenges & learning outcomes (checklist item 6)
│   └── Video_Presentation_Script.docx                          # Script used for the video presentation
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### Option A — Google Colab (recommended, zero setup)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/Financial_Forecasting_Frontier_Distributed_ML.ipynb` (File → Upload notebook).
3. Run all cells (Runtime → Run all). The first code cell installs PySpark automatically.
4. When prompted in the "Data Acquisition" cell, upload `data/bank.csv` (only needed once per session).

### Option B — Local Jupyter
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Financial_Forecasting_Frontier_Distributed_ML.ipynb
```
Make sure `bank.csv` is placed at `data/bank.csv` relative to wherever you launch Jupyter from (this repo is
already laid out that way).

### Option C — Plain script
```bash
pip install -r requirements.txt
python scripts/bank_distributed_ml.py
```

## 📊 Results Summary

| Model | AUC | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.9027 | 0.8806 | 0.8552 |
| Random Forest (100 trees) | 0.9175 | 0.8795 | 0.8438 |

Random Forest was selected as the primary model for its higher AUC, which is the more reliable metric given the
dataset's class imbalance (~88% "no", ~12% "yes").

## 📄 Full Report

See [`docs/Financial_Forecasting_Frontier_Report.docx`](docs/Financial_Forecasting_Frontier_Report.docx) for the
complete write-up covering every part of the pipeline, all 24 analysis charts with explanations, methodology,
challenges faced, and learnings.

## 🛠️ Tech Stack

Hadoop (HDFS concepts) · Hive (SQL warehouse) · Apache Spark · Spark MLlib · Spark Structured Streaming ·
pandas · matplotlib · seaborn · scikit-learn

## 📚 Dataset

`bank.csv` — Bank Marketing Dataset, 4,521 records, 17 attributes (age, job, marital status, education, account
balance, loan status, contact history, campaign details, and subscription outcome).

## 📝 License

This project is submitted as academic coursework for the AlmaBetter Masters in Data Science program.
