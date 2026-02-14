import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================= LOAD DATA =================
# Load Gujarat house price dataset
data = pd.read_csv("../dataset/gujarat_house_price_.csv")
print(" Dataset Loaded :", data.shape)

# ================= CLEANING =================
# Remove missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Standardize column names
data.rename(columns={
    "location": "Location",
    "area_sqft": "Area",
    "bhk": "BHK",
    "bath": "Bathroom",
    "price": "Price"
}, inplace=True)

# ================= OUTLIER REMOVAL =================
# Remove extreme price outliers using IQR
Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1
data = data[
    (data["Price"] >= Q1 - 1.5 * IQR) &
    (data["Price"] <= Q3 + 1.5 * IQR)
]

print(" After Cleaning :", data.shape)

# ================= ENCODING =================
# Encode categorical 'Location' column
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Log transform target variable to reduce skewness
data["Price"] = np.log1p(data["Price"])

# ================= FEATURES & TARGET =================
X = data[["Area", "BHK", "Bathroom", "Location"]]
y = data["Price"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= LINEAR REGRESSION MODEL =================
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Feature scaling
    ("lr", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

# ================= RANDOM FOREST MODEL =================
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=22,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ================= HYBRID PREDICTION =================
# Weighted average of Linear Regression & Random Forest predictions
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_model.predict(X_test)
hybrid_pred = (0.4 * lr_pred) + (0.6 * rf_pred)

# Convert predictions back from log scale
y_test_real = np.expm1(y_test)
hybrid_real = np.expm1(hybrid_pred)

# ================= MODEL EVALUATION =================
r2 = r2_score(y_test, hybrid_pred)
mae = mean_absolute_error(y_test_real, hybrid_real)
rmse = np.sqrt(mean_squared_error(y_test_real, hybrid_real))

print("\nHYBRID MODEL PERFORMANCE")
print("R2 Score        :", round(r2, 3))
print("MAE (₹)         :", round(mae, 2))
print("RMSE (₹)        :", round(rmse, 2))

# 5-fold cross-validation for Linear Regression
cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring="r2")
print("CV R2 Avg       :", round(cv_scores.mean(), 3))

# ================= SAVE MODELS =================
# Save trained models & encoder for deployment
pickle.dump(lr_pipeline, open("linear_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(le, open("location_encoder.pkl", "wb"))

print("\n Models & encoder saved successfully")
