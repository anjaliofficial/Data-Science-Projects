import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/stroke_data.csv")
df.drop(columns=['id'], inplace=True)
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Encode categorical columns
cat_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features & target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Save feature order
joblib.dump(list(X.columns), "models/feature_names.pkl")

# Handle imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Scale
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
joblib.dump(scaler, "models/scaler.pkl")

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
model.fit(X_res_scaled, y_res)
joblib.dump(model, "models/xgb_model.pkl")

print("✅ Training complete. Model, scaler, and features saved.")
