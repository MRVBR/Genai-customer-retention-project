import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap

# =========================
# Load Dataset
# =========================
df = pd.read_csv("Telco-Customer-Churn.csv")

# Drop ID
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions & metrics
y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Save artifacts
pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(explainer, open("shap_explainer.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("encoder_columns.pkl", "wb"))
pickle.dump(metrics, open("model_metrics.pkl", "wb"))

print("✅ Training complete")
print("✅ Model, SHAP explainer, columns & metrics saved")
