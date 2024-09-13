import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import seaborn as sns

file_name = "radar_electricity_data-demo.csv"

def load_data_from_csv(file_name):
    df = pd.read_csv(file_name)
    df = create_labels(df)
    return df

def create_labels(df):
    df['anomaly_label'] = np.where(df['anomaly_consumption'] - df['electricity_consumption'] > 10, 1, 0)
    return df

# Load the data
df = load_data_from_csv(file_name)
X = df[['radar_signal', 'electricity_consumption']]
y = df['anomaly_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='r')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

metrics = ['Accuracy', 'Precision', 'F1 Score', 'ROC-AUC']
scores = [accuracy, precision, f1, roc_auc]

plt.barh(metrics, scores, color=['blue', 'green', 'orange', 'purple'])
plt.title('Model Performance Metrics')
plt.xlabel('Score')
plt.xlim(0, 1)
for i, v in enumerate(scores):
    plt.text(v + 0.01, i, f"{v:.4f}", va='center')
plt.show()

print("\nClassification Report:\n", classification_report(y_test, y_pred))
