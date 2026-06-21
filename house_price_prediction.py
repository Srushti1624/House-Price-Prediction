# House Price Prediction using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create sample housing dataset
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Size": np.random.randint(800, 4000, n),
    "Bedrooms": np.random.randint(1, 6, n),
    "Location": np.random.randint(1, 4, n),   # 1=Rural,2=Suburb,3=City
})

# Target (House Price)
df["Price"] = (
    50000
    + df["Size"] * 150
    + df["Bedrooms"] * 10000
    + df["Location"] * 50000
    + np.random.randint(-20000, 20000, n)
)

# Add some missing values
df.loc[np.random.choice(df.index, 10), "Size"] = np.nan

# -------------------------------
# Explore Dataset
# -------------------------------
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Distribution
plt.hist(df["Price"], bins=20)
plt.title("House Price Distribution")
plt.show()

# -------------------------------
# Handle Missing Data
# -------------------------------
df["Size"].fillna(df["Size"].mean(), inplace=True)

# -------------------------------
# Features and Target
# -------------------------------
X = df[["Size", "Bedrooms", "Location"]]
y = df["Price"]

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# -------------------------------
# Visualization
# -------------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.show()
