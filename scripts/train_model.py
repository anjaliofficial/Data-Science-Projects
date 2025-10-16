# scripts/train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# ---------------------------------------------
# STEP 1: Define paths
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLEAN_FILE = os.path.join(DATA_DIR, "stroke_cleaned.csv")
RAW_FILE = os.path.join(DATA_DIR, "stroke_data.csv")

# ---------------------------------------------
# STEP 2: Clean raw dataset
# ---------------------------------------------
def clean_data():
    """Cleans raw stroke_data.csv and saves stroke_cleaned.csv"""
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"❌ Cannot find {RAW_FILE}. Please place stroke_data.csv inside 'data/' folder.")

    df = pd.read_csv(RAW_FILE)
    print(f"✅ Loaded raw data: {df.shape}")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.drop_duplicates(inplace=True)

    drop_cols = [col for col in ["id", "patient_id"] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Handle Missing BMI
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Drop 'Other' gender and any remaining NaNs
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Other']
    
    df = df.dropna()

    df.to_csv(CLEAN_FILE, index=False)
    print(f"✅ Cleaned data saved to: {CLEAN_FILE} ({df.shape[0]} rows remaining)")
    return df

# ---------------------------------------------
# STEP 3: Load dataset
# ---------------------------------------------
if os.path.exists(CLEAN_FILE):
    print(f"✅ Using existing cleaned data: {CLEAN_FILE}")
    df = pd.read_csv(CLEAN_FILE)
else:
    print("⚠️ stroke_cleaned.csv not found — cleaning raw data now...")
    df = clean_data()

print("Columns in dataset:", df.columns.tolist())

# ---------------------------------------------
# STEP 4: Prepare data
# ---------------------------------------------
target_column = "stroke"
if target_column not in df.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found in dataset.")

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# STEP 5: Train-Test Split (Stratified)
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Important for imbalanced data
)

# ---------------------------------------------
# STEP 6: Train Model
# ---------------------------------------------
print("\n⚙️ Training RandomForestClassifier with balanced weights...")
model = RandomForestClassifier(
    random_state=42, 
    class_weight='balanced' # Important for imbalanced data
) 
model.fit(X_train, y_train)
print("✅ Training complete.")

# ---------------------------------------------
# STEP 7: Evaluate Model
# ---------------------------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------
# STEP 8: Save Model Artifacts
# ---------------------------------------------
model_path = os.path.join(MODEL_DIR, "stroke_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), feature_names_path)

print(f"\n✅ Model saved successfully: {model_path}")
print(f"✅ Scaler saved: {scaler_path}")
print(f"✅ Feature names saved: {feature_names_path}")