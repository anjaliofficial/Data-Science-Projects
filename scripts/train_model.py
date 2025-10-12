import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Auto-detect CSV
possible_paths = [
    "scripts/stroke_cleaned.csv",
    "stroke_cleaned.csv",
    "data/stroke_cleaned.csv"
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    raise FileNotFoundError("Cannot find stroke_cleaned.csv. Place it in scripts/ or root folder.")

df = pd.read_csv(csv_path)

# Fill missing BMI
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Define features
feature_cols = ['gender','age','hypertension','heart_disease','ever_married',
                'work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['stroke']

# Save feature order for SHAP/streamlit
feature_order = X.columns.tolist()

# Scale numeric
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Save artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_order, "models/feature_names.pkl")

print("✅ Training complete. Model, scaler, and features saved.")
