# eda_credit_score.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("credit_data.csv")

# Basic info
print(df.info())
print(df.describe())

# Class balance
sns.countplot(x='default', data=df)
plt.title('Class Balance: Default')
plt.savefig('plots/class_balance.png')

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
