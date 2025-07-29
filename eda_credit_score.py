# eda_credit_score.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure the plots directory exists
os.makedirs("plots", exist_ok=True)

# Load the data
df = pd.read_csv("Score.csv")

# Basic info
print(df.info())
print(df.describe())

# Class distribution of Credit Score
plt.figure(figsize=(8, 5))
sns.countplot(x='Credit_Score', data=df)
plt.title('Distribution of Credit Score Classes')
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/credit_score_distribution.png')
plt.close()

# Print class balance
print(df['Credit_Score'].value_counts())

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

