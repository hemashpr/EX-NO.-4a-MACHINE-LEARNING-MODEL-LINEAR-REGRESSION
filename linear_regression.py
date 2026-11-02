import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Create a reproducible synthetic house-price dataset.
rng = np.random.default_rng(42)
n_samples = 400

square_feet = rng.integers(550, 3200, n_samples)
num_rooms = np.clip(
    np.rint(square_feet / 430 + rng.normal(0, 0.8, n_samples)), 1, 8
).astype(int)
age = rng.integers(0, 41, n_samples)
distance = np.round(rng.uniform(0.5, 35.0, n_samples), 2)

noise = rng.normal(0, 180000, n_samples)
price = (
    450000
    + square_feet * 260
    + num_rooms * 95000
    - age * 8500
    - distance * 12500
    + noise
)
price = np.maximum(price, 350000)

df = pd.DataFrame({
    "square_feet": square_feet,
    "num_rooms": num_rooms,
    "age": age,
    "distance_to_city(km)": distance,
    "price": np.round(price, 2)
})

print("FIRST FIVE RECORDS")
print(df.head().to_string(index=False))
print("\\nDATASET SHAPE:", df.shape)
print("\\nMISSING VALUES")
print(df.isnull().sum().to_string())

# 2. Remove extreme target values using the 1st and 99th percentiles.
lower_limit = df["price"].quantile(0.01)
upper_limit = df["price"].quantile(0.99)
df = df[df["price"].between(lower_limit, upper_limit)].copy()

# 3. Define input features and target; split into training and testing sets.
feature_names = [
    "square_feet",
    "num_rooms",
    "age",
    "distance_to_city(km)"
]
X = df[feature_names]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Fit the scaler on training data only to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\\nTRAINING SET:", X_train.shape)
print("TESTING SET :", X_test.shape)

# 4. Define and train the regression models.
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1000.0, max_iter=20000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
    ),
    "SVR": SVR(kernel="rbf", C=100000, gamma="scale", epsilon=50000),
    "KNN": KNeighborsRegressor(n_neighbors=7)
}

# Polynomial Regression: degree-2 feature expansion followed by Linear Regression.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

predictions = {
    "Polynomial Regression": poly_model.predict(X_test_poly)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions[name] = model.predict(X_test_scaled)

# 5. Evaluate each model using RMSE, MAE and R-squared.
rows = []
for name, y_pred in predictions.items():
    rows.append({
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    })

results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

print("\\nMODEL PERFORMANCE")
print(results.to_string(
    index=False,
    formatters={
        "RMSE": "{:,.2f}".format,
        "MAE": "{:,.2f}".format,
        "R2": "{:.4f}".format
    }
))

best_model = results.iloc[0]["Model"]
print("\\nBEST MODEL BY RMSE:", best_model)

# 6. Generate model-comparison graphs.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
rmse_order = results.sort_values("RMSE", ascending=True)

axes[0].barh(rmse_order["Model"], rmse_order["RMSE"])
axes[0].invert_yaxis()
axes[0].set_title("RMSE Comparison")
axes[0].set_xlabel("RMSE (lower is better)")
axes[0].grid(axis="x", alpha=0.25)

axes[1].barh(rmse_order["Model"], rmse_order["R2"])
axes[1].invert_yaxis()
axes[1].set_title("R-squared Comparison")
axes[1].set_xlabel("R-squared (higher is better)")
axes[1].grid(axis="x", alpha=0.25)

fig.suptitle("House Price Prediction: Regression Model Performance", fontsize=15)
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=180, bbox_inches="tight")
fig.savefig("model_comparison.svg", bbox_inches="tight")
plt.close(fig)

# 7. Plot actual versus predicted prices for the best model.
best_prediction = predictions[best_model]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, best_prediction, alpha=0.7)
minimum = min(y_test.min(), best_prediction.min())
maximum = max(y_test.max(), best_prediction.max())
ax.plot([minimum, maximum], [minimum, maximum], "r--", linewidth=2)
ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
ax.set_title(f"Actual vs Predicted: {best_model}")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("actual_vs_predicted.png", dpi=180, bbox_inches="tight")
fig.savefig("actual_vs_predicted.svg", bbox_inches="tight")
plt.close(fig)

print("\\nOUTPUT IMAGES SAVED:")
print("model_comparison.png")
print("actual_vs_predicted.png")
print("\\nRESULT: Regression models trained and evaluated successfully.")
