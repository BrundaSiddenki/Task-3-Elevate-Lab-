import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("archive\Housing.csv")

# Convert categorical features into numerical values
df.replace({'yes': 1, 'no': 0, 'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0}, inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=['price'])
y = df['price']

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict results
y_pred_linear = linear_model.predict(X_test_scaled)

# Apply Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation Metrics
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"🔹 Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, R² Score: {r2_linear}")
print(f"🔹 Polynomial Regression - MAE: {mae_poly}, MSE: {mse_poly}, R² Score: {r2_poly}")

# Feature Importance Analysis (Linear Model)
coefficients = pd.DataFrame(linear_model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Coefficients:\n", coefficients)

# 📌 Dashboard - Combined Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Creating 4 subplots (2x2)

# 📍 1. Area vs Price
sns.scatterplot(x=X_test['area'], y=y_test, ax=axes[0, 0], color='blue', label="Actual Price")
sns.lineplot(x=X_test['area'], y=y_pred_linear, ax=axes[0, 0], color='red', label="Linear Predicted")
sns.lineplot(x=X_test['area'], y=y_pred_poly, ax=axes[0, 0], color='green', label="Polynomial Predicted")
axes[0, 0].set_title("Area vs Price")

# 📍 2. Bedrooms vs Price
sns.scatterplot(x=X_test['bedrooms'], y=y_test, ax=axes[0, 1], color='green', label="Actual Price")
sns.lineplot(x=X_test['bedrooms'], y=y_pred_linear, ax=axes[0, 1], color='orange', label="Linear Predicted")
sns.lineplot(x=X_test['bedrooms'], y=y_pred_poly, ax=axes[0, 1], color='purple', label="Polynomial Predicted")
axes[0, 1].set_title("Bedrooms vs Price")

# 📍 3. Bathrooms vs Price
sns.scatterplot(x=X_test['bathrooms'], y=y_test, ax=axes[1, 0], color='purple', label="Actual Price")
sns.lineplot(x=X_test['bathrooms'], y=y_pred_linear, ax=axes[1, 0], color='yellow', label="Linear Predicted")
sns.lineplot(x=X_test['bathrooms'], y=y_pred_poly, ax=axes[1, 0], color='cyan', label="Polynomial Predicted")
axes[1, 0].set_title("Bathrooms vs Price")

# 📍 4. Parking vs Price
sns.scatterplot(x=X_test['parking'], y=y_test, ax=axes[1, 1], color='black', label="Actual Price")
sns.lineplot(x=X_test['parking'], y=y_pred_linear, ax=axes[1, 1], color='blue', label="Linear Predicted")
sns.lineplot(x=X_test['parking'], y=y_pred_poly, ax=axes[1, 1], color='magenta', label="Polynomial Predicted")
axes[1, 1].set_title("Parking vs Price")

plt.tight_layout()
plt.show()

# 📌 Residual Analysis - Error Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred_linear, kde=True, bins=30, color="blue")
plt.xlabel("Residuals (Error)")
plt.ylabel("Density")
plt.title("Residual Error Distribution - Linear Model")
plt.show()

# 📌 Heatmap - Correlation between Features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()