#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Lug 07 11:05:49 2025

@author: modal
"""

# %%

import argparse
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

data_path = Path(args.data_path).resolve()
output_path = Path(args.output_path).resolve()

print(f"Loading data from {data_path / 'ashrae_db2.01.csv'}")
try:
    df = pd.read_csv(data_path / "ashrae_db2.01.csv")
except UnicodeDecodeError:
    df = pd.read_csv(data_path / "ashrae_db2.01.csv", encoding="ISO-8859-1")
except Exception as e:
    traceback.print_exc()
    raise ImportError("Loading failed")

# Check the missing values
print("Checking for missing values...")
missing_values = df.isnull().sum()
print(missing_values)

# Percentage of missing values.
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

data1 = df[df["Cooling startegy_building level"] == "Air Conditioned"]
data2 = df[df["Cooling startegy_building level"] == "Naturally Ventilated"]
data3 = df[df["Cooling startegy_building level"] == "Mixed Mode"]
data4 = df[df["Cooling startegy_building level"] == "Mechanically Ventilated"]

df = pd.concat([data1, data2, data3, data4], axis=0)
use_column = "Thermal sensation"

# Making the range from [-3,3] to [-2,2]
df[use_column] = df[use_column].apply(lambda x: -2 if x <= -2 else x)
df[use_column] = df[use_column].apply(lambda x: 2 if x >= 2 else x)

# Rounding off the values to make it categorical in nature
df[use_column] = df[use_column].apply(lambda x: np.round(x))

# Handling NaNs before converting to integer
df = df.dropna(subset=[use_column])  # Remove rows where use_column is NaN

# Convert use_column to integer type
df[use_column] = df[use_column].astype(int)

# Removing NaN values from 'Sex' column as well
df = df.dropna(subset=["Sex"])

# Checking unique values
print(df[use_column].unique())

duplicates_count = df.duplicated().sum()

df = df.drop_duplicates()

# Define numerical as float64 and categorical as int64 or object
numerical_columns = df.select_dtypes(include=["float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["int64", "object"]).columns.tolist()

# Print results
print("Numerical Columns (float64):", numerical_columns)
print("Categorical Columns (int64 & object):", categorical_columns)

# Initialize SimpleImputer with strategy='most_frequent'
imputer = SimpleImputer(strategy="most_frequent")

# Apply imputer to categorical columns
print("Applying SimpleImputer to categorical columns...", end=" ")
df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
print("Done.")

# Initialize KNN Imputer
num_imputer = KNNImputer()

# Apply KNN Imputer only on numerical columns
print("Applying KNNImputer to numerical columns...", end=" ")
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])
print("Done.")

# Combine numerical and encoded categorical features
df = pd.concat([df[numerical_columns], df[categorical_columns]], axis=1)

# Initialize StandardScaler
scaler = StandardScaler()

# Apply StandardScaler only on numerical columns (Ensure they are numeric)
print("Scaling numerical columns...", end=" ")
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print("Done.")

# Ensure categorical column names are valid
categorical_columns = [col for col in categorical_columns if col in df.columns]

# Convert categorical columns to string and fill NaN with 'Unknown'
df[categorical_columns] = df[categorical_columns].astype(str).fillna("Unknown")

# Initialize LabelEncoders dictionary
label_encoders = {}

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for future decoding if needed

output_path.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path / "ASHRAE_processed.csv", index=False)
print(
    f"Data preprocessing completed successfully and saved in {output_path / 'ASHRAE_processed.csv'}!"
)

# %%
