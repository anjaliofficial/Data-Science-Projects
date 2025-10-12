import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("scripts/stroke_cleaned.csv")  # Make sure you have cleaned CSV

# Fill missing numeric values
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Encode categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

# Features and target
X = df_encoded.drop("stroke", axis=1)
y = df_encoded["stroke"]

# Save feature order
feature_order = X.columns.tolist()
joblib.dump(feature_order, "models/feature_names.pkl")
print("✅ Feature names saved.")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

joblib.dump(scaler, "models/scaler.pkl")
print("✅ Scaler saved.")

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/xgb_model.pkl")
print("✅ Model trained and saved.")

# Evaluate
score = model.score(X_test, y_test)
print(f"✅ Test Accuracy: {score*100:.2f}%")
