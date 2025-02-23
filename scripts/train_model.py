import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

# Load dataset
df = pd.read_csv("data/fruit_data_with_colors.txt", delimiter="\t")

# Split features and target
X = df[['mass', 'width', 'height', 'color_score']]
y = df['fruit_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = round(accuracy_score(y_test, y_pred), 4)
precision = round(precision_score(y_test, y_pred, average="weighted"), 4)
recall = round(recall_score(y_test, y_pred, average="weighted"), 4)

# Save model and scaler
joblib.dump(model, "backend/models/fruit_classifier.joblib")
joblib.dump(scaler, "backend/models/scaler.joblib")

# Save metrics
metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
with open("backend/models/metrics.json", "w") as f:
    json.dump(metrics, f)

print("Model and Scaler retrained successfully!")
print(json.dumps(metrics, indent=4))  # Print metrics for logging
