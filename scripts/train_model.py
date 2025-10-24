import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import numpy as np

# ---------------------------------------------
# STEP 1: Define paths and setup directories
# ---------------------------------------------
# Define paths relative to the script's location (assuming the script is in 'scripts/')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data and Model directories are now defined one level up from the script's location,
# to match your project structure: PROJECT_ROOT/data/ and PROJECT_ROOT/models/
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLEAN_FILE = os.path.join(DATA_DIR, "stroke_cleaned.csv")
RAW_FILE = os.path.join(DATA_DIR, "stroke_data.csv") # Path: PROJECT_ROOT/data/stroke_data.csv

# ---------------------------------------------
# STEP 2: Clean raw dataset
# ---------------------------------------------
def clean_data():
    """Cleans raw stroke_data.csv, handles NAs, drops irrelevant columns, and saves stroke_cleaned.csv"""
    
    print(f"Attempting to load raw data from: {RAW_FILE}")
    if not os.path.exists(RAW_FILE):
        # Raising the error immediately if the file is not found at the expected path.
        raise FileNotFoundError(f"❌ Cannot find {RAW_FILE}. Please place stroke_data.csv inside the **'data/'** folder in your project root.")

    # --- FIX APPLIED HERE: Use the explicit RAW_FILE path ---
    df = pd.read_csv(RAW_FILE)
    print(f"✅ Loaded raw data: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("residence_type", "residence_type")
    
    df.drop_duplicates(inplace=True)

    # Robustly drop ID columns
    drop_cols = [col for col in ["id", "patient_id"] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Handle 'Other' gender and any remaining NaNs
    if 'gender' in df.columns:
        # Drop the 'Other' gender as it's typically very rare and complicates OHE
        df = df[df['gender'].str.lower() != 'other']
    
    # Handle Missing BMI by imputation (median)
    if "bmi" in df.columns:
        # Ensure 'N/A' strings are converted to actual NaN before imputation
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce') 
        bmi_median = df["bmi"].median()
        df["bmi"] = df["bmi"].fillna(bmi_median)
        print(f"Imputed {df['bmi'].isna().sum()} missing BMI values with median ({bmi_median:.2f}).")
    
    # Drop rows with any remaining missing values (should only be from other columns if any)
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows due to remaining NaN values.")


    df.to_csv(CLEAN_FILE, index=False)
    print(f"✅ Cleaned data saved to: {CLEAN_FILE} ({df.shape[0]} rows remaining)")
    return df

# ---------------------------------------------
# STEP 3: Load dataset (Clean if necessary)
# ---------------------------------------------
if os.path.exists(CLEAN_FILE):
    try:
        df = pd.read_csv(CLEAN_FILE)
        # Check if the cleaned file is valid
        if df.empty or 'stroke' not in df.columns:
            print("⚠️ Existing cleaned file is invalid. Cleaning raw data now...")
            df = clean_data()
        else:
            print(f"✅ Using existing cleaned data: {CLEAN_FILE}")
    except Exception:
        print("⚠️ Error reading cleaned file. Cleaning raw data now...")
        df = clean_data()
else:
    print("⚠️ stroke_cleaned.csv not found — cleaning raw data now...")
    df = clean_data()

print("Columns in dataset:", df.columns.tolist())

# ---------------------------------------------
# STEP 4: Prepare data (OHE and Scaling)
# ---------------------------------------------
target_column = "stroke"
if target_column not in df.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found in dataset.")

# Identify categorical columns (must match the model logic in prediction)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Crucial for consistency: drop_first=True to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

# Check for zero variance in features (important if the clean data is very small)
if X.empty:
    raise ValueError("❌ Feature matrix X is empty after encoding and cleaning.")

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# STEP 5: Train-Test Split (Stratified)
# ---------------------------------------------
# Stratification is critical due to the severe class imbalance in stroke data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
)

# ---------------------------------------------
# STEP 6: Train Model
# ---------------------------------------------
print("\n⚙️ Training RandomForestClassifier with balanced weights...")
# Use class_weight='balanced' to handle the imbalanced dataset
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10, # Limiting depth can help with overfitting slightly
    random_state=42, 
    class_weight='balanced' 
) 
model.fit(X_train, y_train)
print("✅ Training complete.")

# ---------------------------------------------
# STEP 7: Evaluate Model
# ---------------------------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=['No Stroke (0)', 'Stroke (1)']))

# ---------------------------------------------
# STEP 8: Save Model Artifacts
# ---------------------------------------------
model_path = os.path.join(MODEL_DIR, "stroke_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
# Save the EXACT feature order (critical for OHE alignment in prediction)
joblib.dump(X.columns.tolist(), feature_names_path) 

print(f"\n✅ Model saved successfully: {model_path}")
print(f"✅ Scaler saved: {scaler_path}")
print(f"✅ Feature names saved: {feature_names_path}")

print("\n--- Training Script Finished ---")
