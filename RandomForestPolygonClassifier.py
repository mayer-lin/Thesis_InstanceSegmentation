#######################################################################################################
#### This script utilizes Random Forest to flag whether a polygon is of an individual or cluster ####
#######################################################################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path =  "data/input" # xlsx sheet. In QGIS, polygon geometries were calculated, and the according attribute table was exported.
df = pd.read_excel(file_path, sheet_name="Sheet1")
df.columns = df.columns.str.strip() # Optional

# Rename and clean. The original names were created by QGIS.
df.rename(columns={"Indv_or_cl": "category_encoded"}, inplace=True)
feature_label_mapping = {
    "area": "Area",
    "perimeter": "Perimeter",
    "roundness": "Roundness",
    "perim2area": "Perimeter-to-Area Ratio",
    "convexhull_ratio": "Convex Hull Ratio",
    "dist2nearest_indiv": "Distance to Nearest Individual"
}
for col in feature_label_mapping.keys():
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
df.rename(columns=feature_label_mapping, inplace=True)
features = list(feature_label_mapping.values())

# Adjust labeled data
df_labeled = df.dropna(subset=features + ["category_encoded"]).copy()
df_labeled["category_encoded"] = df_labeled["category_encoded"].str.strip().map({"Indv": 0, "Cl": 1}) # Encodes Individual as 0 and Cluster as 1
X_labeled = df_labeled[features]
y_labeled = df_labeled["category_encoded"].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.1, random_state=42) # Update percentage for testing as needed. Maintain a fixed random state.

# Train model with calibration
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight={0: 1, 1: 1}) # Adjust as needed
calibrated_rf = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)

# Thresholding logic. Essentially, if not 95% sure it's an individual then label as cluster.
individual_prob = calibrated_rf.predict_proba(X_test)[:, 0]
y_pred = (individual_prob < 0.95).astype(int)

# Evaluate on training dataset
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report on Testing Data:\n", classification_report(y_test, y_pred))

# Predict on the labeled subset
full_indiv_prob = calibrated_rf.predict_proba(df_labeled[features])[:, 0]
df_labeled["cluster_prob"] = 1 - full_indiv_prob
df_labeled["individual_prob"] = full_indiv_prob
df_labeled["predicted_soft"] = (full_indiv_prob < 0.95).astype(int)
df_labeled["Ground Truth"] = y_labeled.values

# Predict on any unlabeled polygons
df_unlabeled = df[df["category_encoded"].isna()].copy()
if not df_unlabeled.empty:
    df_unlabeled = df_unlabeled.dropna(subset=features)
    unlabeled_indiv_prob = calibrated_rf.predict_proba(df_unlabeled[features])[:, 0]
    df_unlabeled["cluster_prob"] = 1 - unlabeled_indiv_prob
    df_unlabeled["individual_prob"] = unlabeled_indiv_prob
    df_unlabeled["predicted_soft"] = (unlabeled_indiv_prob < 0.95).astype(int)
    print("Predictions made for unlabeled polygons.")
else:
    print("No unlabeled polygons found.")

# Combine labeled and unlabeled back into one DataFrame
df_combined = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

# Save the combined results
df_combined.to_excel("___/results.xlsx", index=False) # Add save location.
print("Results saved to _____ ")