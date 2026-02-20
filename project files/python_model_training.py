import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("motor_temp.csv")

# Features and target
X = data.drop("stator_temp", axis=1)
y = data["stator_temp"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))

# Save model
joblib.dump(model, "motor_model.pkl")
joblib.dump(scaler, "scaler.pkl")