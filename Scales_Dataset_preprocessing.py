#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Lug 07 14:53:11 2025

@author: modal
"""

# %%
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import date, time
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

# %%

parser = argparse.ArgumentParser(
    description="Preprocessing for Hierarchical Transformer with Multi-Head Self-Attention for Sensor Data"
)
parser.add_argument(
    "--data_path", type=str, default="./data", help="Path to the dataset CSV file"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./preprocessed_data",
    help="Path to save the processed dataset",
)
args, _ = parser.parse_known_args()

file_path = Path(args.data_path).resolve() / "Scales_Dataset.csv"
output_path = Path(args.output_path).resolve()
output_path.mkdir(parents=True, exist_ok=True)
output_file = output_path / "Scales_Dataset_processed.csv"

# %%

df = pd.read_csv(file_path, encoding="ISO-8859-1")

# %%
# Check the missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Identify columns with >50% missing
columns_above_50_missing = missing_percentage[missing_percentage > 50].sort_values(
    ascending=False
)

# Drop these columns from the dataframe
df.drop(columns=columns_above_50_missing.index, axis=1, inplace=True)

# %%

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


# %%
# Display the data types of all columns
print(df.dtypes)

# %%
# Identify potential date/time columns based on column names
date_time_cols = [
    col for col in df.columns if "date" in col.lower() or "time" in col.lower()
]

# Convert date columns to date only, and time columns to time only
conversion_results = {}
for col in date_time_cols:
    try:
        df[col] = pd.to_datetime(
            df[col], errors="coerce"
        )  # First convert to full datetime
        if "date" in col.lower():
            df[col] = df[col].dt.date  # Keep only date part
            conversion_results[col] = "Converted to date format"
        elif "time" in col.lower():
            df[col] = df[col].dt.time  # Keep only time part
            conversion_results[col] = "Converted to time format"
    except Exception as e:
        conversion_results[col] = f"Failed to convert: {str(e)}"

# %%

# Re-identify column types
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Correctly identify date and time columns
date_columns = [
    col for col in df.columns if df[col].apply(lambda x: isinstance(x, date)).any()
]
time_columns = [
    col for col in df.columns if df[col].apply(lambda x: isinstance(x, time)).any()
]

# Remove identified date/time columns from categorical list if present
categorical_columns = [
    col for col in categorical_columns if col not in date_columns + time_columns
]

# Display the results
column_types_summary = {
    "Numerical Columns": numerical_columns,
    "Categorical Columns": categorical_columns,
    "Date Columns": date_columns,
    "Time Columns": time_columns,
}


# %%

# Step 1: Drop rows with missing target values
df = df.dropna(subset=["StateTSVF"])

# Step 2: Encode original 1–7 values to labels 0–6
le = LabelEncoder()
df["StateTSVF"] = le.fit_transform(df["StateTSVF"])  # Now 0 to 6

# Step 3: Map from label 0–6 to -3 to +3
initial_map = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}
df["StateTSVF"] = df["StateTSVF"].map(initial_map)

# ✅ Step 4: Map from -3 to +3 into 5-point scale (0 to 4)
five_point_map = {-3: 0, -2: 1, -1: 1, 0: 2, 1: 3, 2: 3, 3: 4}
df["StateTSVF"] = df["StateTSVF"].map(five_point_map)

# Step 5: Prepare numeric data
X = df.select_dtypes(include=["int64", "float64"]).fillna(df.mean(numeric_only=True))
y = df["StateTSVF"]

# Step 6: Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 7: Check class distribution
print("Original distribution:\n", y.value_counts().sort_index())
print("Resampled distribution:\n", pd.Series(y_resampled).value_counts().sort_index())


# %%
data1 = df[df["StatusHVACcat"] == "Air conditioned"]
data2 = df[df["StatusHVACcat"] == "Natural Ventilation"]
data3 = df[df["StatusHVACcat"] == "Mixed ventilation"]
data4 = df[df["StatusHVACcat"] == "Mech ventilation"]
data5 = df[df["StatusHVACcat"] == "No HVAC"]
data6 = df[df["StatusHVACcat"] == "Heated"]
data7 = df[df["StatusHVACcat"] == "unknown"]
data8 = df[df["StatusHVACcat"].isna()]
df = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8], axis=0)


# %%

# Fill categorical missing values with mode.
for col in categorical_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Use KNNImputer for numerical columns.
knn_imputer = KNNImputer(n_neighbors=5)
df[numerical_columns] = knn_imputer.fit_transform(df[numerical_columns])

# %%
# Drop all rows with any missing values from the DataFrame
df = df.dropna()

# Check remaining shape of the dataset
df.shape

# %%
# Temporal feature extraction
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["SurvDate"] = pd.to_datetime(df["SurvDate"], errors="coerce")
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time
df["SurvTimeStart"] = pd.to_datetime(
    df["SurvTimeStart"], format="%H:%M:%S", errors="coerce"
).dt.time
df["SurvTimeEnd"] = pd.to_datetime(
    df["SurvTimeEnd"], format="%H:%M:%S", errors="coerce"
).dt.time

df["date_month"] = df["Date"].dt.month
df["date_day"] = df["Date"].dt.day
df["date_weekday"] = df["Date"].dt.weekday
df["surv_month"] = df["SurvDate"].dt.month
df["surv_day"] = df["SurvDate"].dt.day
df["surv_weekday"] = df["SurvDate"].dt.weekday


def time_to_minutes(t):
    if pd.isnull(t):
        return np.nan
    return t.hour * 60 + t.minute + t.second / 60


df["time_minutes"] = df["Time"].apply(time_to_minutes)
df["surv_start_minutes"] = df["SurvTimeStart"].apply(time_to_minutes)
df["surv_end_minutes"] = df["SurvTimeEnd"].apply(time_to_minutes)
df["surv_duration"] = df["surv_end_minutes"] - df["surv_start_minutes"]

df.drop(
    columns=[
        "Index",
        "SurvID",
        "unID",
        "Date",
        "Time",
        "SurvDate",
        "SurvTimeStart",
        "SurvTimeEnd",
    ],
    inplace=True,
)

# %%
# Convert to integer type
df["StateTSVF"] = df["StateTSVF"].astype(int)

# Confirm it's now integer
print(df["StateTSVF"].unique())  # Should show array of ints
print(df["StateTSVF"].dtype)  # Should show 'int64'

df["StateTSVF"] = pd.Categorical(df["StateTSVF"])


# %%

# Include 'StateTSVF' in categorical_columns if it should be treated as a label
categorical_columns = (
    categorical_columns + ["StateTSVF"]
    if "StateTSVF" not in categorical_columns
    else categorical_columns
)

# Remove non-existent columns from the list
categorical_columns = [col for col in categorical_columns if col in df.columns]
numerical_columns = [
    col
    for col in numerical_columns
    if col in df.columns and col not in categorical_columns
]

# Ensure all values in categorical columns are strings before encoding
for col in categorical_columns:
    df[col] = df[col].astype(str).fillna("missing")

# Apply Label Encoding to categorical columns (including StateTSVF)
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Apply StandardScaler to numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# %%
# Save the updated DataFrame to a CSV file
df.to_csv(output_file, index=False)

# %%
