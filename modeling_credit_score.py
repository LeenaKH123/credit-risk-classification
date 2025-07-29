# modeling_credit_score.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Step 1: Load and Prepare Data
# -----------------------------
df = pd.read_csv("Score.csv")
print("Columns:", df.columns.tolist())

# Encode the target variable: Credit_Score (e.g., Poor, Standard, Good)
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'])

# Drop categorical variables that aren't numerically encoded
exclude_cols = ['Payment_of_Min_Amount', 'Credit_Mix', 'Payment_Behaviour']
df = df.drop(columns=exclude_cols, errors='ignore')

# Separate features and target
X = df.drop(columns='Credit_Score')
y = df['Credit_Score']

# -----------------------------
# Step 2: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 3: Train Random Forest
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predict and Evaluate
# -----------------------------
y_pred = model.predict(X_test)

# Convert numeric labels back to original class names
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Print evaluation metrics
print("Confusion Matrix:")
cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)

print("\nClassification Report:")
report = classification_report(y_test_labels, y_pred_labels)
print(report)

# -----------------------------
# Step 5: Save Results to File
# -----------------------------
os.makedirs("plots", exist_ok=True)

with open("plots/classification_report.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# -----------------------------
# Step 6: Feature Importance Plot
# -----------------------------
importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("plots/feature_importance.png")
plt.show()

# -----------------------------
# Step 7: Confusion Matrix Heatmap
# -----------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("plots/confusion_matrix_heatmap.png")
plt.show()
