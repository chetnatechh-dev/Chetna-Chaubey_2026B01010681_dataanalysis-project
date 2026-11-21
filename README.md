# Chetna-Chaubey_2026B01010681_dataanalysis-project
Data cleaning project using Python and pandas to handle missing values, duplicates and formatting errors.
This repository contains a cleaned dataset of student academic performance, suitable for Python-based exploratory data analysis, visualization, and predictive modeling. The dataset includes demographic, academic, and lifestyle-related variables for 1,000 students.


DATASET OVERVIEW
File: Clean_Students_Performance_Dataset.csv
Format: CSV
Primary key: STUDENT ID
Rows: 1,000 student records
Intended use: Educational analytics, feature engineering, correlation analysis, classification, and regression modeling

The cleaned dataset includes both original academic and lifestyle fields and engineered features such as AVG SCORE, STUDY SLEEP RATIO, and SCORE IMPROVEMENT.


LOADING THE DATASET
import pandas as pd

df = pd.read_csv("Clean_Students_Performance_Dataset.csv")

# Inspect the data
print(df.shape)
print(df.info())
print(df.head())


BASIC DATA CHECKS
# Check missing values
print(df.isnull().sum())

# Check duplicate student IDs
print(df["STUDENT ID"].duplicated().sum())

# Summary statistics
print(df.describe())

# View categorical distributions
print(df["DEPARTMENT"].value_counts())
print(df["GENDER"].value_counts())
print(df["FAMILY INCOME LEVEL"].value_counts())


SUGGESTED ANALYSIS
Compare AVG SCORE across departments, gender, family income level, and internet access.
Examine relationships between attendance, study hours, sleep, stress, and academic performance.
Analyze SCORE IMPROVEMENT to identify students whose final scores improved or declined relative to midterm performance.
Visualize score distributions using histograms, box plots, and correlation heatmaps.
Build models to predict AVG SCORE or classify students into performance groups.


EXAMPLE VISUALISATION
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="DEPARTMENT", y="AVG SCORE")
plt.title("Average Score by Department")
plt.xlabel("Department")
plt.ylabel("Average Score")
plt.tight_layout()
plt.show()


EXAMPLE CORRELATION ANALYSIS
numeric_cols = [
    "ATTENDANCE (%)",
    "MIDTERM SCORE",
    "FINAL SCORE",
    "ASSIGNMENTS AVG",
    "QUIZZES AVG",
    "STUDY HOURS PER WEEK",
    "STRESS LEVEL (1-10)",
    "SLEEP HOURS PER NIGHT",
    "AVG SCORE",
    "STUDY SLEEP RATIO",
    "SCORE IMPROVEMENT"
]

correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix["AVG SCORE"].sort_values(ascending=False))


NOTES
Personal identifiers such as names and email addresses should not be used as model features.
Treat STUDENT ID as an identifier rather than a predictive variable.
Categorical variables such as DEPARTMENT, GENDER, EXTRACURRICULAR ACTIVITIES, INTERNET ACCESS AT HOME, and FAMILY INCOME LEVEL should be encoded before machine-learning modeling.
SCORE IMPROVEMENT can be negative, indicating that a student's final score was lower than their midterm score.
